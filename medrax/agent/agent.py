import json
import operator
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage, AIMessage

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
logging.basicConfig(level=logging.INFO)

_ = load_dotenv()


class ToolCallLog(TypedDict):
    """
    A TypedDict representing a log entry for a tool call.

    Attributes:
        timestamp (str): The timestamp of when the tool call was made.
        tool_call_id (str): The unique identifier for the tool call.
        name (str): The name of the tool that was called.
        args (Any): The arguments passed to the tool.
        content (str): The content or result of the tool call.
    """

    timestamp: str
    tool_call_id: str
    name: str
    args: Any
    content: str


class AgentState(TypedDict):
    """
    A TypedDict representing the state of an agent.

    Attributes:
        messages (Annotated[List[AnyMessage], operator.add]): A list of messages
            representing the conversation history. The operator.add annotation
            indicates that new messages should be appended to this list.
    """

    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    """
    A class representing an agent that processes requests and executes tools based on
    language model responses.

    Attributes:
        model (BaseLanguageModel): The language model used for processing.
        tools (Dict[str, BaseTool]): A dictionary of available tools.
        checkpointer (Any): Manages and persists the agent's state.
        system_prompt (str): The system instructions for the agent.
        workflow (StateGraph): The compiled workflow for the agent's processing.
        log_tools (bool): Whether to log tool calls.
        log_path (Path): Path to save tool call logs.
    """

    def __init__(
        self,
        model: BaseLanguageModel,
        tools: List[BaseTool],
        intent_recognition_model: str = "Qwen/Qwen3-0.6B", # add by Bryce
        checkpointer: Any = None,
        system_prompt: str = "",
        log_tools: bool = True,
        log_dir: Optional[str] = "logs",
    ):
        """
        Initialize the Agent.

        Args:
            model (BaseLanguageModel): The language model to use.
            tools (List[BaseTool]): A list of available tools.
            checkpointer (Any, optional): State persistence manager. Defaults to None.
            system_prompt (str, optional): System instructions. Defaults to "".
            log_tools (bool, optional): Whether to log tool calls. Defaults to True.
            log_dir (str, optional): Directory to save logs. Defaults to 'logs'.
        """
        self.system_prompt = system_prompt
        self.log_tools = log_tools

        if self.log_tools:
            self.log_path = Path(log_dir or "logs")
            self.log_path.mkdir(exist_ok=True)

        # Define the agent workflow
        workflow = StateGraph(AgentState)
        workflow.add_node("process", self.process_request)
        workflow.add_node("execute", self.execute_tools)
        workflow.add_conditional_edges(
            "process", self.has_tool_calls, {True: "execute", False: END}
        )
        workflow.add_edge("execute", "process")
        workflow.set_entry_point("process")
        self.workflow = workflow.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

        # Add by Bryce
        self.intent_recognition_tokenizer = AutoTokenizer.from_pretrained(intent_recognition_model)
        self.intent_recognition_model = AutoModelForCausalLM.from_pretrained(
            intent_recognition_model,
            torch_dtype="auto",
            device_map="auto"
        )
        
    def process_request(self, state: AgentState) -> Dict[str, List[AnyMessage]]:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """

        state = self.preprocess_request(state)  # Preprocess the request to recognize intent

        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        response = self.model.invoke(messages)
        logging.info(f"Process node output: {response}")

        return {"messages": [response]}

    def has_tool_calls(self, state: AgentState) -> bool:
        """
        Check if the response contains any tool calls.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            bool: True if tool calls exist, False otherwise.
        """
        response = state["messages"][-1]
        return len(response.tool_calls) > 0

    def execute_tools(self, state: AgentState) -> Dict[str, List[ToolMessage]]:
        """
        Execute tool calls from the model's response.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[ToolMessage]]: A dictionary containing tool execution results.
        """
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for call in tool_calls:
            print(f"Executing tool: {call}")
            if call["name"] not in self.tools:
                print("\n....invalid tool....")
                result = "invalid tool, please retry"
            else:
                result = self.tools[call["name"]].invoke(call["args"])

            results.append(
                ToolMessage(
                    tool_call_id=call["id"],
                    name=call["name"],
                    args=call["args"],
                    content=str(result),
                )
            )

        self._save_tool_calls(results)
        print("Returning to model processing!")

        return {"messages": results}

    def _save_tool_calls(self, tool_calls: List[ToolMessage]) -> None:
        """
        Save tool calls to a JSON file with timestamp-based naming.

        Args:
            tool_calls (List[ToolMessage]): List of tool calls to save.
        """
        if not self.log_tools:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.log_path / f"tool_calls_{timestamp}.json"

        logs: List[ToolCallLog] = []
        for call in tool_calls:
            log_entry = {
                "tool_call_id": call.tool_call_id,
                "name": call.name,
                "args": call.args,
                "content": call.content,
                "timestamp": datetime.now().isoformat(),
            }
            logs.append(log_entry)

        with open(filename, "w") as f:
            json.dump(logs, f, indent=4)

    def preprocess_request(self, state: AgentState) -> AgentState:
        """
        Process the request using the language model.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            Dict[str, List[AnyMessage]]: A dictionary containing the model's response.
        """

        user_input = state["messages"][-1]

        if isinstance(user_input, ToolMessage) or isinstance(user_input, AIMessage):
            return state

        print("user_input:", user_input)

        user_text_query = self.extract_text_content(user_input)
        user_image_query = self.extract_image_content(state["messages"])

        user_intent = self.recognize_intent(user_text_query)
        print("user_intent: ", user_intent)

        processed_user_query = f"The user's query: {user_text_query}\n{user_intent}"
        next(item.update({"text": processed_user_query}) for item in state["messages"][-1]['content'] if item.get("type") == "text")

        return state


    def extract_text_content(self, input_dict):
        """
        Extract the text content for type 'text' and the text for type 'image' if present.

        Args:
            input_dict (dict): The input dictionary containing 'content' list.

        Returns:
            tuple: A tuple containing:
                - The text content for type 'text' (str).
                - The text content for type 'image' if present, otherwise None.
        """
        text_content = None

        for item in input_dict.get('content', []):
            if item.get('type') == 'text' and text_content is None:
                text_content = item.get('text')

        return text_content

    def extract_image_content(self, messages):
        """
        Reverse iterate through the list to find the first ToolMessage class and check for image_path.

        Args:
            messages (list): The list of messages to process.

        Returns:
            dict or None: The dictionary containing image_path if found, otherwise None.
        """
        tool_message_index = None

        # Find the first ToolMessage instance in reverse order
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                tool_message_index = i
                break
        
        # If no ToolMessage is found, set index to the start of the list
        if tool_message_index is None:
            tool_message_index = 0

        # Iterate from the last element to the ToolMessage instance (or start of the list)
        for i in range(len(messages) - 1, tool_message_index - 1, -1):
            message = messages[i]
            if isinstance(message, dict) and "content" in message:
                content = message["content"]
                if isinstance(content, str) and "image_path:" in content:
                    return message  # Return the message containing 'image_path'

        return None

    def recognize_intent(self, query: str) -> str:
        """
        Recognize the intent of a user query using the Qwen-3 model.

        Args:
            query (str): The user input query.

        Returns:
            str: The recognized intent.
        """
        # Tokenize the input query
        SYSTEM_PROMPT = """
        You are an AI assistant specialized in identifying user intents within the domain of dentistry.
        Note that user intent refers to the goal or purpose the user wants to achieve through interacting with the large language model, 
        not just the literal meaning of the user’s input. Please ensure you deeply understand the user’s needs and accurately extract their core intent.
        Given any user input, including questions, instructions, or descriptions, your task is to accurately recognize and generate the underlying intent.
        
        Possible user intent include, but are not limited to:
        • Visual description (e.g., describing visual apparence or conditions of dental images)
        • Disease diagnosis
        • Subtyping, staging, grading, and classification of conditions
        • Treatment planning
        • Prognosis prediction
        • Disease prevention
        • Dental knowledge look‑up
        • Other oral‑medicine–related intents

        Always identify the user’s intent clearly and unambiguously. Your output must be follow for format of the following sentence:
        The user’s intent: <YOUR OUTPUT>. 

        Note that the output should only contain the intent without any additional explanations or answers to users' questions.
        """

        text = self.intent_recognition_tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},  # System prompt
                {"role": "user", "content": query}  # User query
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        model_inputs = self.intent_recognition_tokenizer([text], return_tensors="pt").to(self.intent_recognition_model.device)

        # Conduct text completion
        generated_ids = self.intent_recognition_model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # Parse the thinking content
        try:
            # Find the index of the </think> token (151668)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # Decode the intent content
        intent_content = self.intent_recognition_tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        return intent_content