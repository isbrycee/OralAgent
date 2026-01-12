"""
MedRAX Application Main Module
This module serves as the entry point for the MedRAX medical imaging AI assistant.
It provides functionality to initialize an AI agent with various medical imaging tools
and launch a web interface for interacting with the system.
The system uses OpenAI's language models for reasoning and can be configured
with different model weights, tools, and parameters.
"""

import os
import warnings
from typing import *
from dotenv import load_dotenv
from transformers import logging

from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI

from interface import create_demo
from medrax.agent import *
from medrax.tools import *
from medrax.utils import *

warnings.filterwarnings("ignore")
logging.set_verbosity_error()
_ = load_dotenv()


def initialize_agent(
    prompt_file,
    tools_to_use=None,
    model_dir="/model-weights",
    temp_dir="temp",
    device="cuda",
    model="chatgpt-4o-latest",
    temperature=0.7,
    top_p=0.95,
    rag_config: Optional[RAGConfig] = None,
    openai_kwargs={}
):
    """Initialize the MedRAX agent with specified tools and configuration.

    Args:
        prompt_file (str): Path to file containing system prompts
        tools_to_use (List[str], optional): List of tool names to initialize. If None, all tools are initialized.
        model_dir (str, optional): Directory containing model weights. Defaults to "/model-weights".
        temp_dir (str, optional): Directory for temporary files. Defaults to "temp".
        device (str, optional): Device to run models on. Defaults to "cuda".
        model (str, optional): Model to use. Defaults to "chatgpt-4o-latest".
        temperature (float, optional): Temperature for the model. Defaults to 0.7.
        top_p (float, optional): Top P for the model. Defaults to 0.95.
        rag_config (RAGConfig, optional): Configuration for the RAG tool. Defaults to None.
        openai_kwargs (dict, optional): Additional keyword arguments for OpenAI API, such as API key and base URL.

    Returns:
        Tuple[Agent, Dict[str, BaseTool]]: Initialized agent and dictionary of tool instances
    """

    # Load system prompts from file
    prompts = load_prompts_from_file(prompt_file)
    prompt = prompts["MEDICAL_ASSISTANT"]
    prompt_for_intent_recognition = prompts["INTENT_RECOGNITION_ASSISTANT"]

    # Define all available tools with their initialization functions
    all_tools = {
        "ChestXRayClassifierTool": lambda: ChestXRayClassifierTool(device=device),
        "ChestXRaySegmentationTool": lambda: ChestXRaySegmentationTool(device=device),
        "LlavaMedTool": lambda: LlavaMedTool(cache_dir=model_dir, device=device, load_in_8bit=True),
        "XRayVQATool": lambda: XRayVQATool(cache_dir=model_dir, device=device),
        "ChestXRayReportGeneratorTool": lambda: ChestXRayReportGeneratorTool(
            cache_dir=model_dir, device=device
        ),
        "XRayPhraseGroundingTool": lambda: XRayPhraseGroundingTool(
            cache_dir=model_dir, temp_dir=temp_dir, load_in_8bit=True, device=device
        ),
        "ChestXRayGeneratorTool": lambda: ChestXRayGeneratorTool(
            model_path=f"{model_dir}/roentgen", temp_dir=temp_dir, device=device
        ),
        "ImageVisualizerTool": lambda: ImageVisualizerTool(),
        "DicomProcessorTool": lambda: DicomProcessorTool(temp_dir=temp_dir),
        
        ###################### add by bryce ######################
        # for Panoramic X-ray modality
        "PanoramicXRayToothIdDetectionTool": lambda: PanoramicXRayToothDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth", 
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.json", 
            temp_dir=temp_dir,
            device=device
        ),
        "PanoramicXRayPeriapicalLesionSubClassDetectionTool": lambda: PanoramicXRayPeriapicalLesionSubClassDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.pth", 
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.json", 
            temp_dir=temp_dir,
            device=device
        ),
        "PanoramicXRayDiseaseSegmentationTool": lambda: PanoramicXRayDiseaseSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.pth", 
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.json", 
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device
        ),
        "PanoramicXRayJawStructureSegmentationTool": lambda: PanoramicXRayJawStructureSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.pth", 
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.json", 
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device
        ),

        # for Periapical X-ray modality
        "PeriapicalXRayDiseaseSegmentationTool": lambda: PeriapicalXRayDiseaseSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.pth", 
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.json", 
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device
        ),

        # for Cephalometric X-ray modality
        "CephalometricXRayLandmarkDetectionTool": lambda: CephalometricXRayLandmarkDetectionTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth", 
            prototype_path=f"{model_dir}/config_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth",  
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.json",
            image_size=(512, 512),
            temp_dir=temp_dir,
            device=device
        ),
        
        ###################### for RAG ######################
        "MedicalRAGTool": lambda: RAGTool(config=rag_config),
    }

    # Initialize only selected tools or all if none specified
    # tools_dict = {}
    tools_dict: Dict[str, BaseTool] = {}
    tools_to_use = tools_to_use or all_tools.keys()
    for tool_name in tools_to_use:
        if tool_name in all_tools:
            tools_dict[tool_name] = all_tools[tool_name]()

    # Set up checkpointing for conversation state
    checkpointer = MemorySaver()
    # Initialize the language model
    model = ChatOpenAI(model=model, temperature=temperature, top_p=top_p, **openai_kwargs)

    # Create the agent with the specified model, tools, and configuration
    agent = Agent(
        model,
        tools=list(tools_dict.values()),
        log_tools=True,
        log_dir="logs",
        system_prompt=prompt,
        intent_recognition_prompt=prompt_for_intent_recognition,
        checkpointer=checkpointer,
    )

    print("Agent initialized")
    return agent, tools_dict


if __name__ == "__main__":
    """
    This is the main entry point for the MedRAX application.
    It initializes the agent with the selected tools and creates the demo.
    """
    print("Starting server...")

    # Example: initialize with only specific tools
    # Here three tools are commented out, you can uncomment them to use them
    selected_tools = [
        "ImageVisualizerTool", # For displaying images in the UI
        # "DicomProcessorTool",
        # "ChestXRayClassifierTool",
        # "ChestXRaySegmentationTool",
        # "ChestXRayReportGeneratorTool",
        # "XRayVQATool",
        # "LlavaMedTool",
        # "XRayPhraseGroundingTool",
        # "ChestXRayGeneratorTool",

        ################## Add by Bryce ##################
        "PanoramicXRayToothIdDetectionTool",
        "PanoramicXRayBoneLossSegmentationTool",
        "PanoramicXRayDiseaseSegmentationTool",
        "PanoramicXRayPeriapicalLesionSubClassDetectionTool",
        "PanoramicXRayJawStructureSegmentationTool",
        "PeriapicalXRayDiseaseSegmentationTool",
        "CephalometricXRayLandmarkDetectionTool"
    

        ################## for RAG ##################
        "MedicalRAGTool", # For retrieval-augmented generation with medical knowledge
    ]


    ################## for RAG ##################
    # Configure the Retrieval Augmented Generation (RAG) system
    # This allows the agent to access and use medical knowledge documents
    rag_config = RAGConfig(
        model="command-a-03-2025",  # Invalid in current version; TODO: support Qwen3 model
        embedding_model="Qwen/Qwen3-Embedding-0.6B", # "4B, 8B"
        rerank_model="Qwen/Qwen3-Reranker-0.6B", # "4B, 8B"
        temperature=0.7,
        persist_dir="medrax/rag/vectorDB",  # Change this to the target path of the vector database
        chunk_size=1000,
        chunk_overlap=100,
        retriever_k=3,
        local_docs_dir="medrax/rag/docs",  # Change this to the path of the documents for RAG
        use_medrag_textbooks=True,  # Set to True if you want to use the MedRAG textbooks dataset
    )

    # Prepare OpenAI API configuration from environment variables
    openai_kwargs: Dict[str, str] = {}
    if api_key := os.getenv("OPENAI_API_KEY"):
        openai_kwargs["api_key"] = api_key

    if base_url := os.getenv("OPENAI_BASE_URL"):
        openai_kwargs["base_url"] = base_url

    # Initialize the agent with all configured components
    agent, tools_dict = initialize_agent(
        "medrax/docs/system_prompts.txt",
        tools_to_use=selected_tools,
        model_dir="/data/OralGPT/OralGPT-expert-model-repository",  # Change this to the path of the model weights
        temp_dir="temp",  # Change this to the path of the temporary directory
        device="cuda",  # Change this to the device you want to use
        model="gpt-5-nano",  # Change this to the model you want to use, e.g. gpt-4o-mini
        temperature=0.7,
        top_p=0.95,
        rag_config=rag_config,
        openai_kwargs=openai_kwargs
    )

    # Create and launch the web interface
    demo = create_demo(agent, tools_dict)

    demo.launch(server_name="0.0.0.0", server_port=8551, share=True)
