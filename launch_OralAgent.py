import warnings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from medrax.agent import Agent
from medrax.utils import load_prompts_from_file
from langgraph.checkpoint.memory import MemorySaver
from medrax.tools import *
from langchain_core.runnables import Runnable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
import uuid

warnings.filterwarnings("ignore")
_ = load_dotenv()

OralAgent = FastAPI()

# 声明全局变量（不再使用全局 thread，每次请求用独立 thread_id 隔离状态）
agent = None

# 定义请求体模型
class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 消息列表，符合 vllm 的输入格式

def get_agent(
    tools,
    prompt_file,
    model_name,
    temperature,
    model_dir,
    device="cuda",
):
    # Load prompts
    prompts = load_prompts_from_file(prompt_file)
    system_prompt = prompts["MEDICAL_ASSISTANT"]
    intent_recognition_prompt = prompts["INTENT_RECOGNITION_ASSISTANT"]
    enriched_query_template = prompts.get("ENRICHED_QUERY_TEMPLATE")
    modality_section_template = prompts.get("MODALITY_SECTION_TEMPLATE")

    # Initialize the agent
    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=0.95)

    intent_classifier_model = BioMedCLIPClassifier(
        checkpoint_path=f"{model_dir}/OralGPT_Modality_Identification_BioMedCLIP_8modalities.pth",
        coco_names_path=f"{model_dir}/categories_Modality_Identification_BioMedCLIP_8modalities.json",
        num_classes=8,
        device=device,
    )

    agent = Agent(
        model,
        intent_classifier_model=intent_classifier_model,
        tools=tools,
        log_tools=True,
        log_dir="logs",
        system_prompt=system_prompt,
        intent_recognition_prompt=intent_recognition_prompt,
        enriched_query_template=enriched_query_template,
        modality_section_template=modality_section_template,
        checkpointer=checkpointer,
    )
    return agent

def run_OralAgent(agent, messages, thread_id: Optional[str] = None):
    """Run agent for one request. Each request uses an isolated thread so state does not leak.
    Pass thread_id only when you want to continue a specific conversation."""
    thread = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}
    final_response = None
    for event in agent.workflow.stream({"messages": messages}, thread):
        for v in event.values():
            final_response = v

    final_response = final_response["messages"][-1].content.strip()
    agent_state = agent.workflow.get_state(thread)

    return final_response, str(agent_state)

@OralAgent.get("/test")
def test_endpoint():
    return {"status": "OralAgent is working"}

@OralAgent.on_event("startup")
async def startup_event():
    global agent
    expert_model_dir = "/data/OralGPT/OralGPT-expert-model-repository"
    temp_dir = "temp"
    # 单进程时用默认 "cuda"；多 worker 时由 gunicorn_conf 的 post_fork 设置 CUDA_VISIBLE_DEVICES，本进程内 cuda:0 即对应分配的那张卡
    device = "cuda"
    model_name = "gpt-5-nano"
    temperature = 0.2

    ROOT = "/home/jinghao/projects/OralGPT-Agent/OralAgent"
    PROMPT_FILE = f"{ROOT}/medrax/docs/system_prompts.txt"

    tools = get_tools(
        model_dir=expert_model_dir,
        temp_dir=temp_dir,
        device=device
    )
    agent = get_agent(
        tools,
        prompt_file=PROMPT_FILE,
        model_name=model_name,
        temperature=temperature,
        model_dir=expert_model_dir,
        device=device,
    )
    print("OralAgent successfully initialized and ready to use.")

    # for route in OralAgent.routes:
    #     print(route.path, route.name)


# 添加 API 路由
@OralAgent.post("/v1/chat/completions")
def run_agent_endpoint(request: ChatCompletionRequest):
    try:
        # 每次请求使用新的 thread_id，多次请求之间状态互不影响
        response, state = run_OralAgent(
            agent=agent,
            messages=request.messages,
        )
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",  # 生成唯一 ID
            "object": "chat.completion",
            "created": int(time.time()),  # 当前时间戳
            "response": response,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response,  # 使用生成的响应内容
                        "annotations": [],
                        "refusal": None,
                    }
                }
            ],

            # "state": state,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # 单进程开发/调试；多卡多 worker 请用: ./run_multi_gpu.sh 或 gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py launch_OralAgent:OralAgent
    uvicorn.run("launch_OralAgent:OralAgent", host="0.0.0.0", port=8124, reload=True)