"""Tool factories and default tool list for MedRAX agent. All tool definitions live here."""
from typing import Callable, Dict, List, Optional, Any

###################### add by bryce ######################
from .panoramic_radiograph.toothIdDetection import *
from .panoramic_radiograph.boneLossSegmentation import *
from .panoramic_radiograph.diseaseSegmentation import *
from .panoramic_radiograph.periapicalLesionSubClassDetection import *
from .panoramic_radiograph.jawStructureSegmentation import *

from .periapical_radiograph.diseaseSegmentation import *
from .periapical_radiograph.abnormality7Classification import *

from .cephalometric_radiograph.cephalometricLandmarkDetection import *

from .intraoral_image.conditionDetection import *
from .intraoral_image.gingivitisDetection import *
from .intraoral_image.fenestrationDetection import *
from .intraoral_image.malocclusionIssuesDetection import *
from .intraoral_image.abnormal9classification import *
from .intraoral_image.toothTypeDetection import *

from .cytopathology.cellNucleusSegmentation import *
from .cytopathology.cellNucleusGrading import *

from .histopathology.OSCCSegmentation import *
from .histopathology.OSCC5Classification import *
from .histopathology.Leukoplakia3Classification import *

###################### for RAG ######################
from .rag import *


# Default list of tool names to use when initializing the agent (single source of truth)
DEFAULT_SELECTED_TOOL_NAMES: List[str] = [
    "PanoramicXRayToothIdDetectionTool",
    "PanoramicXRayBoneLossSegmentationTool",
    "PanoramicXRayDiseaseSegmentationTool",
    "PanoramicXRayPeriapicalLesionSubClassDetectionTool",
    "PanoramicXRayJawStructureSegmentationTool",
    "PeriapicalXRayDiseaseSegmentationTool",
    "PeriapicalXRayAbnormalityClassificationTool",
    "CephalometricXRayLandmarkDetectionTool",
    "IntraoralImageConditionDetectionTool",
    "IntraoralImageGingivitisDetectionTool",
    "IntraoralImageFenestrationDetectionTool",
    "IntraoralImageMalocclusionIssuesDetectionTool",
    "IntraoralImageImageLevelConditionDetectionTool",
    "IntraoralImageToothTypeDetectionTool",
    "CytopathologyCellNucleusSegmentationTool",
    "CytopathologyCellNucleusGradingTool",
    "HistopathologyLeukoplakiaOSCCClassificationTool",
    "HistopathologyOSMFOSCCClassificationTool",
    "HistopathologyOSCCSegmentationTool",
    "MedicalRAGTool",
]


def get_all_tools_factories(
    model_dir: str,
    temp_dir: str,
    device: str,
    rag_config: Optional[Any] = None,
) -> Dict[str, Callable[[], Any]]:
    """
    Return a dict of tool_name -> factory (callable that returns an initialized tool instance).
    Used by main.py to instantiate only the selected tools.

    Args:
        model_dir: Path to model weights and configs.
        temp_dir: Temporary directory for intermediate files.
        device: Device (e.g. "cuda", "cpu").
        rag_config: Optional RAGConfig for MedicalRAGTool.

    Returns:
        Dict mapping tool name to a no-arg callable that returns the tool instance.
    """
    # Panoramic X-ray
    def f_panoramic_tooth():
        return PanoramicXRayToothDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_panoramic_x-ray_32ToothID.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_panoramic_bone_loss():
        return PanoramicXRayBoneLossSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_1disease_boneLoss.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_1disease_boneLoss.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_1disease_boneLoss.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    def f_panoramic_periapical_lesion():
        return PanoramicXRayPeriapicalLesionSubClassDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_r50_panoramic_x-ray_3subclasses_periapicalLesion.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_panoramic_disease_seg():
        return PanoramicXRayDiseaseSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_11diseases.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    def f_panoramic_jaw_structure():
        return PanoramicXRayJawStructureSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_panoramic_x-ray_2structures_mandibularCanal_maxillarySinus.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    # Periapical X-ray
    def f_periapical_disease_seg():
        return PeriapicalXRayDiseaseSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_SwinL_periapical_x-ray_6diseases.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    def f_periapical_abnormality():
        return PeriapicalXRayAbnormal7ClassificationTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Periapical_7abnormalities.safetensors",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Periapical_7abnormalities.json",
            temp_dir=temp_dir,
            device=device,
        )

    # Cephalometric X-ray
    def f_cephalometric_landmark():
        return CephalometricXRayLandmarkDetectionTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth",
            prototype_path=f"{model_dir}/config_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_CeLDA_UNet2D_cephalometric_x-ray_29Landmarks.json",
            image_size=(512, 512),
            temp_dir=temp_dir,
            device=device,
        )

    # Intraoral Image (use intraoral_image in paths, consistent with main.py)
    def f_intraoral_condition():
        return IntraoralImageConditionDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_intraoral_gingivitis():
        return IntraoralImageGingivitisDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_intraoral_fenestration():
        return IntraoralImageFenestrationDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_intraoral_tooth_type():
        return IntraoralImageToothTypeDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_intraoral_malocclusion():
        return IntraoralImageMalocclusionIssuesDetectionTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.py",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_intraoral_image_level():
        return IntraoralImageAbnormal9ClassificationTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_intraoral_image_9conditions.safetensors",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_intraoral_image_9conditions.json",
            temp_dir=temp_dir,
            device=device,
        )

    # Cytopathology
    def f_cytopathology_seg():
        return CytopathologyCellNucleusSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    def f_cytopathology_grading():
        return CytopathologyCellNucleusGradingTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    # Histopathology
    def f_histopathology_oscc_seg():
        return HistopathologyOSCCSegmentationTool(
            config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.yaml",
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.pth",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.json",
            confidence_threshold=0.3,
            temp_dir=temp_dir,
            device=device,
        )

    def f_histopathology_osmf_oscc():
        return HistopathologyOSCC5ClassificationTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_OSMF_OSCC_5conditions.safetensors",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_OSMF_OSCC_5conditions.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_histopathology_leukoplakia():
        return HistopathologyLeukoplakia3ClassificationTool(
            checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_Leukoplakia_OSCC_3diseases.safetensors",
            coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_Leukoplakia_OSCC_3diseases.json",
            temp_dir=temp_dir,
            device=device,
        )

    def f_rag():
        if rag_config is None:
            raise ValueError("MedicalRAGTool requires rag_config to be passed to get_all_tools_factories.")
        return RAGTool(config=rag_config)

    return {
        "PanoramicXRayToothIdDetectionTool": f_panoramic_tooth,
        "PanoramicXRayBoneLossSegmentationTool": f_panoramic_bone_loss,
        "PanoramicXRayPeriapicalLesionSubClassDetectionTool": f_panoramic_periapical_lesion,
        "PanoramicXRayDiseaseSegmentationTool": f_panoramic_disease_seg,
        "PanoramicXRayJawStructureSegmentationTool": f_panoramic_jaw_structure,
        "PeriapicalXRayDiseaseSegmentationTool": f_periapical_disease_seg,
        "PeriapicalXRayAbnormalityClassificationTool": f_periapical_abnormality,
        "CephalometricXRayLandmarkDetectionTool": f_cephalometric_landmark,
        "IntraoralImageConditionDetectionTool": f_intraoral_condition,
        "IntraoralImageGingivitisDetectionTool": f_intraoral_gingivitis,
        "IntraoralImageFenestrationDetectionTool": f_intraoral_fenestration,
        "IntraoralImageToothTypeDetectionTool": f_intraoral_tooth_type,
        "IntraoralImageMalocclusionIssuesDetectionTool": f_intraoral_malocclusion,
        "IntraoralImageImageLevelConditionDetectionTool": f_intraoral_image_level,
        "CytopathologyCellNucleusSegmentationTool": f_cytopathology_seg,
        "CytopathologyCellNucleusGradingTool": f_cytopathology_grading,
        "HistopathologyOSCCSegmentationTool": f_histopathology_oscc_seg,
        "HistopathologyOSMFOSCCClassificationTool": f_histopathology_osmf_oscc,
        "HistopathologyLeukoplakiaOSCCClassificationTool": f_histopathology_leukoplakia,
        "MedicalRAGTool": f_rag,
    }


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
    periapical_abnormality_classification_tool = PeriapicalXRayAbnormal7ClassificationTool(
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Periapical_7abnormalities.safetensors",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Periapical_7abnormalities.json",
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

    # Intraoral Image tools (paths use intraoral_image)
    intraoral_condition_detection_tool = IntraoralImageConditionDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_4conditions.json",
        temp_dir=temp_dir,
        device=device
    )
    intraoral_gingivitis_detection_tool = IntraoralImageGingivitisDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_gingivitis.json",
        temp_dir=temp_dir,
        device=device
    )
    intraoral_fenestration_detection_tool = IntraoralImageFenestrationDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_fenestration.json",
        temp_dir=temp_dir,
        device=device
    )
    intraoral_tooth_type_detection_tool = IntraoralImageToothTypeDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_8ToothTypes.json",
        temp_dir=temp_dir,
        device=device
    )
    intraoral_malocclusion_issues_detection_tool = IntraoralImageMalocclusionIssuesDetectionTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.py",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINO_SwinL_5scale_intraoral_image_9malocclusionIssues.json",
        temp_dir=temp_dir,
        device=device
    )
    intraoral_image_level_condition_detection_tool = IntraoralImageAbnormal9ClassificationTool(
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_intraoral_image_9conditions.safetensors",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_intraoral_image_9conditions.json",
        temp_dir=temp_dir,
        device=device
    )

    # Cytopathology tools
    cytopathology_cell_nucleus_segmentation_tool = CytopathologyCellNucleusSegmentationTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Cytopathology_7conditions.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )
    cytopathology_cell_nucleus_grading_tool = CytopathologyCellNucleusGradingTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Cytopathology_4gradings.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )

    # Histopathology tools
    histopathology_oscc_segmentation_tool = HistopathologyOSCCSegmentationTool(
        config_path=f"{model_dir}/config_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.yaml",
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.pth",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_MaskDINO_r50_Histopathology_OSCC_Segmentation.json",
        confidence_threshold=0.3,
        temp_dir=temp_dir,
        device=device
    )
    histopathology_osmf_oscc_classification_tool = HistopathologyOSCC5ClassificationTool(
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_OSMF_OSCC_5conditions.safetensors",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_OSMF_OSCC_5conditions.json",
        temp_dir=temp_dir,
        device=device
    )
    histopathology_leukoplakia_oscc_classification_tool = HistopathologyLeukoplakia3ClassificationTool(
        checkpoint_path=f"{model_dir}/OralGPT_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_Leukoplakia_OSCC_3diseases.safetensors",
        coco_names_path=f"{model_dir}/categories_Visual_Expert_Model_DINOv3_ImageLevel_Histopathology_Leukoplakia_OSCC_3diseases.json",
        temp_dir=temp_dir,
        device=device
    )

    rag_config = RAGConfig(
        model="command-a-03-2025",  # Invalid in current version; TODO: support Qwen3 model
        embedding_model="Qwen/Qwen3-Embedding-8B",  # "0.6B, 4B, 8B"
        rerank_model="Qwen/Qwen3-Reranker-8B",  # "0.6B, 4B, 8B"
        temperature=0.7,
        retriever_k=3,
        persist_dir="medrax/rag/",  # Base path for vector DB; subdir added when use_OralCorpus=True
        use_OralCorpus=True,  # Set to True to load Oral corpus when creating vectorstore
        corpus_language="chinese",  # "english" -> vectorDB_OralCorpus_English, "chinese" -> vectorDB_OralCorpus_Chinese
        local_docs_dir="",  # Path to Oral corpus (EN or CN) for RAG; also used for custom docs
        chunk_size=1000,  # Only valid for private documents
        chunk_overlap=100,  # Only valid for private documents
    )
    
    # RAG tool
    # medical_rag_tool = RAGTool(config=rag_config)

    return [
        panoramic_tooth_detection_tool,
        panoramic_periapical_lesion_tool,
        panoramic_disease_segmentation_tool,
        panoramic_jaw_structure_tool,
        periapical_disease_segmentation_tool,
        periapical_abnormality_classification_tool,
        cephalometric_landmark_detection_tool,
        intraoral_condition_detection_tool,
        intraoral_gingivitis_detection_tool,
        intraoral_fenestration_detection_tool,
        intraoral_tooth_type_detection_tool,
        intraoral_malocclusion_issues_detection_tool,
        intraoral_image_level_condition_detection_tool,
        cytopathology_cell_nucleus_segmentation_tool,
        cytopathology_cell_nucleus_grading_tool,
        histopathology_oscc_segmentation_tool,
        histopathology_osmf_oscc_classification_tool,
        histopathology_leukoplakia_oscc_classification_tool,
        # medical_rag_tool,
    ]