import warnings
from dotenv import load_dotenv
from langserve import add_routes
from langchain_openai import ChatOpenAI
from medrax.agent import Agent
from medrax.utils import load_prompts_from_file
from langgraph.checkpoint.memory import MemorySaver
from medrax.tools import *
from langchain_core.runnables import Runnable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel
import time
import uuid

warnings.filterwarnings("ignore")
_ = load_dotenv()

OralAgent = FastAPI()

# 声明全局变量
agent = None
thread = None

# 定义请求体模型
class ChatCompletionRequest(BaseModel):
    messages: List[Dict[str, Any]]  # 消息列表，符合 vllm 的输入格式

def get_agent(tools, prompt_file, model_name, temperature):
    # Load prompts
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]

    # Initialize the agent
    checkpointer = MemorySaver()
    model = ChatOpenAI(model=model_name, temperature=temperature, top_p=0.95)
    agent = Agent(
        model,
        tools=tools,
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        checkpointer=checkpointer,
    )
    thread = {"configurable": {"thread_id": "1"}}
    return agent, thread

# def run_OralAgent(agent, thread, prompt, image_urls=[]):
#     messages = [
#         HumanMessage(
#             content=[
#                 {"type": "text", "text": prompt},
#             ]
#             + [{"type": "image_url", "image_url": {"url": image_url}} for image_url in image_urls]
#         )
#     ]

#     final_response = None
#     for event in agent.workflow.stream({"messages": messages}, thread):
#         for v in event.values():
#             final_response = v

#     final_response = final_response["messages"][-1].content.strip()
#     agent_state = agent.workflow.get_state(thread)

#     return final_response, str(agent_state)

def run_OralAgent(agent, thread, messages):
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
    global agent, thread
    expert_model_dir = "/data/OralGPT/OralGPT-expert-model-repository"
    temp_dir = "temp"
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
    agent, thread = get_agent(tools, prompt_file=PROMPT_FILE, model_name=model_name, temperature=temperature)
    print("OralAgent successfully initialized and ready to use.")

    for route in OralAgent.routes:
        print(route.path, route.name)


# 添加 API 路由
@OralAgent.post("/v1/chat/completions")
def run_agent_endpoint(request: ChatCompletionRequest):
    try:
        response, state = run_OralAgent(
            agent=agent,
            thread=thread,
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
    # # Define the paths and device
    # expert_model_dir = "/data/OralGPT/OralGPT-expert-model-repository"  # Replace with the actual path to the model directory
    # temp_dir = "temp"   # Replace with the actual path to the temporary directory
    # device = "cuda"     # Replace with "cuda" or "cpu" based on your setup
    # model_name = "gpt-5-nano"
    # temperature = 0.2

    # # Setup directory paths
    # ROOT = "/home/jinghao/projects/OralGPT-Agent/OralAgent"  # Set this directory to where MedRAX is
    # PROMPT_FILE = f"{ROOT}/medrax/docs/system_prompts.txt"

    # # Initialize tools
    # tools = get_tools(
    #     model_dir=expert_model_dir,
    #     temp_dir=temp_dir,
    #     device=device
    # )
    
    # # Start the agent
    # agent, thread = get_agent(tools, prompt_file=PROMPT_FILE, model_name=model_name, temperature=temperature)
    # print("Agent successfully initialized and ready to use.")

    # for route in OralAgent.routes:
    #     print(route.path, route.name)

    import uvicorn
    uvicorn.run("launch_OralAgent:OralAgent", host="0.0.0.0", port=8124, reload=True)