###################### add by bryce ######################
from .panoramic_radiograph.toothIdDetection import *
from .panoramic_radiograph.boneLossSegmentation import *
from .panoramic_radiograph.diseaseSegmentation import *
from .panoramic_radiograph.periapicalLesionSubClassDetection import *
from .panoramic_radiograph.jawStructureSegmentation import *

from .periapical_radiograph.diseaseSegmentation import *

from .cephalometric_radiograph.cephalometricLandmarkDetection import *

###################### for RAG ######################
from .rag import *


def get_tools(
    model_dir: str,
    temp_dir: str,
    device: str
):
    """
    Initialize and return a list of tools.

    Args:
        model_dir (str): Path to the directory containing model weights and configurations.
        temp_dir (str): Path to the temporary directory for intermediate files.
        device (str): Device to use for computation (e.g., "cuda" or "cpu").

    Returns:
        list: A list of initialized tools.
    """
    # Panoramic X-ray tools
    panoramic_tooth_detection_tool = PanoramicXRayToothDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.json",
        temp_dir=temp_dir,
        device=device
    )
    panoramic_periapical_lesion_tool = PanoramicXRayPeriapicalLesionSubClassDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.json",
        temp_dir=temp_dir,
        device=device
    )
    panoramic_disease_segmentation_tool = PanoramicXRayDiseaseSegmentationTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )
    panoramic_jaw_structure_tool = PanoramicXRayJawStructureSegmentationTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )

    # Periapical X-ray tools
    periapical_disease_segmentation_tool = PeriapicalXRayDiseaseSegmentationTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )

    # Cephalometric X-ray tools
    cephalometric_landmark_detection_tool = CephalometricXRayLandmarkDetectionTool(
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth",
        prototype_path=f"{model_dir}/config_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.json",
        image_size=(512, 512),
        temp_dir=temp_dir,
        device=device
    )

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

    # RAG tool
    medical_rag_tool = RAGTool(config=rag_config)

    return [
        panoramic_tooth_detection_tool,
        panoramic_periapical_lesion_tool,
        panoramic_disease_segmentation_tool,
        panoramic_jaw_structure_tool,
        periapical_disease_segmentation_tool,
        cephalometric_landmark_detection_tool,
        medical_rag_tool
    ]