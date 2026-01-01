from typing import Dict, List, Optional, Tuple, Type, Any
from pathlib import Path
import uuid
import tempfile

import numpy as np
import torch
import torchvision
import torchxrayvision as xrv
import matplotlib.pyplot as plt
import skimage.io
import skimage.measure
import skimage.transform
import traceback

from pydantic import BaseModel, Field
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


############################### for MaskDINO Segmentation Model ##################################
import os
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from ..model_MaskDINO import add_maskdino_config
from ..model_MaskDINO.predictor import VisualizationDemo
from typing import Any, Dict, List, Optional, Type
import time
import json
import random
from detectron2.data import Metadata


class PanoramicXRayBoneLossSegmentationInput(BaseModel):
    """Input schema for the Panoramic X-ray Bone Loss Segmentation Tool."""
    image_path: str = Field(..., description="Path to the Panoramic X-ray image file to be processed")

class PanoramicXRayBoneLossSegmentationOutput(BaseModel):
    """Output schema for MaskDINO Bone Loss Segmentation Tool."""
    segments: List[Dict[str, Any]] = Field(..., description="List of segmented regions with their properties")


class PanoramicXRayBoneLossSegmentationTool(BaseTool):
    """Tool for performing bone loss segmentation analysis of panoramic X-ray images using MaskDINO."""

    name: str = "panoramic_xray_bone_loss_segmentation"
    description: str = (
        "Detects and segments regions of bone loss in panoramic X-ray images."
    )

    args_schema: Type[BaseModel] = PanoramicXRayBoneLossSegmentationInput
    temp_dir: Path = Path("temp")
    cfg: Any = None
    demo: Any = None
    device: Optional[str] = "cuda"
    coco_names_path: str = ""
    id2name: Any = None
    category_metadata: Any = None

    def __init__(self, config_path: str, checkpoint_path: str, coco_names_path: str, confidence_threshold: float = 0.3, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
        """Initialize the MaskDINO Bone Loss Segmentation Tool."""
        super().__init__()
        self.cfg = self._setup_cfg(config_path, checkpoint_path, confidence_threshold)
        self.coco_names_path = coco_names_path
        self.id2name = self._load_category_names()
        self.category_metadata = self._load_category_metadata('panoramic_X-ray_1disease_boneLoss')
        self.demo = VisualizationDemo(self.cfg, self.category_metadata)
        self.device = device
        self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
    def _setup_cfg(self, config_path: str, checkpoint_path: str, confidence_threshold: float):
        """Set up the configuration for MaskDINO."""
        cfg = get_cfg()
        cfg.confidence_threshold = confidence_threshold
        add_deeplab_config(cfg)
        add_maskdino_config(cfg)
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = checkpoint_path

        cfg.freeze()
        return cfg


    def _run(
        self,
        image_path: str,
        diseases: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Run segmentation analysis for specified diseases (currently only supports 'bone_loss')."""
        try:
            # Validate and get disease indices
            if diseases:
                diseases = [d.strip() for d in diseases]
                invalid_diseases = [d for d in diseases if d != "bone_loss"]
                if invalid_diseases:
                    raise ValueError(f"Invalid diseases specified: {invalid_diseases}")
            else:
                diseases = ["bone_loss"]
        
            # Read the image
            img = read_image(image_path, format="BGR")

            # Run inference
            start_time = time.time()
            predictions, visualized_output = self.demo.run_on_image(img, self.cfg.confidence_threshold)
            inference_time = time.time() - start_time

            # Process predictions
            results = []
            if "instances" in predictions:
                instances = predictions["instances"].to("cpu")
                masks = instances.pred_masks.numpy()
                scores = instances.scores.numpy()
                labels = instances.pred_classes.numpy()

                for mask, score, label in zip(masks, scores, labels):
                    category_name = self.id2name[label]
                    if category_name not in diseases:
                        continue
                    
                    # 获取外接四边形
                    y_indices, x_indices = np.where(mask)  # 获取 mask 中非零像素的坐标
                    if len(y_indices) > 0 and len(x_indices) > 0:
                        x_min, x_max = x_indices.min(), x_indices.max()
                        y_min, y_max = y_indices.min(), y_indices.max()
                        bbox = [x_min, y_min, x_max, y_max]  # [xmin, ymin, xmax, ymax]
                    else:
                        bbox = None  # 如果 mask 为空，则返回默认值

                    if bbox is not None:
                        results.append({
                            "label": category_name,
                            "bbox": bbox,
                            "score": round(float(score), 2)
                        })

            print(f"results: {results}")

            # Save visualization
            viz_path = os.path.join(self.temp_dir, f"visualized_{uuid.uuid4().hex[:8]}.png")
            os.makedirs(os.path.dirname(viz_path), exist_ok=True)
            visualized_output.save(viz_path)
            print(f"Visualization saved to: {viz_path}")

            # Prepare output and metadata
            output = {
                "segmentation_image_path": viz_path,
                "results": results,
            }

            metadata = {
                "image_path": image_path,
                "segmentation_image_path": viz_path,
                "inference_time": inference_time,
                "original_size": img.shape[:2],
                "analysis_status": "completed",
            }

            return output, metadata

        except Exception as e:
            error_output = {"error": str(e)}
            error_metadata = {
                "image_path": image_path,
                "analysis_status": "failed",
                "error_traceback": traceback.format_exc(),
            }
            return error_output, error_metadata
            
    async def _arun(
        self,
        image_path: str,
        confidence: float,
        diseases: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Tuple[Dict[str, Any], Dict]:
        """Async version of _run."""
        return self._run(image_path, confidence, diseases)

    def _load_category_names(self):
        """Load category names from COCO format."""
        with open(self.coco_names_path, 'r') as f:
            categories = json.load(f)
        return {int(cat_id): cat_name for cat_id, cat_name in categories.items()}

    def _load_category_metadata(self, name):
        # 随机生成颜色
        random.seed(42)  # 固定随机种子，确保每次生成的颜色一致
        thing_classes = list(self.id2name.values())
        thing_colors = [
            [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            for _ in thing_classes
        ]

        # 创建 Metadata 对象
        metadata = Metadata(name=name)
        metadata.thing_classes = thing_classes
        metadata.thing_colors = thing_colors

        return metadata

    # def _preprocess_image(self, image_path: str):
    #         """Preprocess the input image."""
    #         # transform = T.Compose([
    #         #     T.RandomResize([800], max_size=1333),
    #         #     T.ToTensor(),
    #         #     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         # ])
    #         transform = Compose([
    #             RandomResize([800], max_size=1333),
    #             ToTensor(),
    #             Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #         ])
    #         image = Image.open(image_path).convert("RGB")
    #         orig_size = image.size
    #         image_transformed, _ = transform(image, None)
    #         return image_transformed, orig_size

    # def _run_inference(self, image_tensor):
    #     """Run inference on the input image tensor."""
    #     with torch.no_grad():
    #         image_tensor = image_tensor.unsqueeze(0).to(self.device)
    #         outputs = self.model(image_tensor)
    #         processed_outputs = self.postprocessors['bbox'](outputs, torch.tensor([[1.0, 1.0]]).to(self.device))[0]
    #     return processed_outputs

    # def _run(self, 
    #         image_path: str,
    #         confidence: Optional[float] = 0.3,
    #         tooth_ids: Optional[List[str]] = None,
    #         run_manager: Optional[CallbackManagerForToolRun] = None,
    #         ) -> PanoramicXRayToothDetectionOutput:
    #     try:
    #         """Run the DINO Tooth Detection Tool."""
    #         # Preprocess the image
    #         image_tensor, orig_size = self._preprocess_image(image_path)
    #         orig_w, orig_h = orig_size

    #         # Run inference
    #         outputs = self._run_inference(image_tensor)

    #         # Process detections
    #         detections = []
    #         for i in range(len(outputs['boxes'])):
    #             box_cxcywh = box_ops.box_xyxy_to_cxcywh(outputs['boxes'][i]).cpu().numpy()
    #             cx, cy, w, h = box_cxcywh
    #             box_orig = [
    #                 cx * orig_w,
    #                 cy * orig_h,
    #                 w * orig_w,
    #                 h * orig_h
    #             ]
    #             label = outputs['labels'][i].item()
    #             score = outputs['scores'][i].item()

    #             if score < confidence:
    #                 continue

    #             tooth_id = self.id2name.get(label, f"Unknown ({label})")
    #             detections.append({
    #                 "tooth_id": tooth_id,
    #                 "bbox": self._convert_bbox_to_xyxy(box_orig),
    #                 "score": round(score, 2)
    #             })

    #         if tooth_ids:
    #             detections = [det for det in detections if det['tooth_id'] in tooth_ids]

    #         # Create output object
    #         output = PanoramicXRayToothDetectionOutput(detections=detections)

    #         # Save visualization
    #         viz_path = self._save_visualization(
    #                 image_path=image_path,
    #                 output=output
    #                 )

    #         # convert boxes for detected teeth
    #         results = {}
    #         for detection in detections:
    #             bbox = detection['bbox']
    #             score = detection['score']
    #             tooth_id = detection['tooth_id']
    #             results[tooth_id] = {'box': bbox, 'score': score}

    #         # Prepare output and metadata
    #         output = {
    #             "detection_image_path": viz_path,
    #             "detections": detections,
    #             # "metrics": {tooth_id: metrics.dict() for tooth_id, metrics in results.items()},
    #         }

    #         metadata = {
    #             "image_path": image_path,
    #             "detection_image_path": viz_path,
    #             "original_size": orig_size,
    #             "model_size": tuple(image_tensor.shape[-2:]),
    #             "confidence_threshold": confidence,
    #             "processed_teeth": list(results.keys()),
    #             "analysis_status": "completed",
    #         }
    #         return output, metadata

    #     except Exception as e:
    #         # Handle errors and prepare error output and metadata
    #         error_output = {"error": str(e)}
    #         error_metadata = {
    #             "image_path": image_path,
    #             "analysis_status": "failed",
    #             "error_traceback": traceback.format_exc(),
    #         }
    #         return error_output, error_metadata

    # async def _arun(
    #     self,
    #     image_path: str,
    #     confidence: float,
    #     tooth_ids: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForToolRun] = None,
    # ) -> Tuple[Dict[str, Any], Dict]:
    #     """Async version of _run."""
    #     return self._run(image_path, confidence, tooth_ids)

    # @staticmethod
    # def _convert_bbox_to_xyxy(box_orig):
    #     """Convert bbox from cx, cy, w, h to xmin, ymin, xmax, ymax."""
    #     cx, cy, w, h = box_orig
    #     x1 = round(float(cx - w / 2))
    #     y1 = round(float(cy - h / 2))
    #     x2 = round(float(cx + w / 2))
    #     y2 = round(float(cy + h / 2))
    #     return [x1, y1, x2, y2]

    # def _save_visualization(self, image_path: str, output: PanoramicXRayToothDetectionOutput) -> str:
    #     """
    #     Visualize the detection results and save the visualization as an image.

    #     Args:
    #         image_path (str): Path to the input image.
    #         output (DINOToothDetectionOutput): Detection output containing tooth IDs, bounding boxes, and scores.
    #     """
    #     # Load the original image
    #     orig_image = Image.open(image_path).convert("RGB")
    #     orig_w, orig_h = orig_image.size

    #     # Create a figure and axis for visualization
    #     fig, ax = plt.subplots(1, figsize=(12, 10))
    #     ax.imshow(orig_image)
    #     ax.axis('off')
    #     ax.set_title(f"Tooth Detection Results: {os.path.basename(image_path)}")

    #     # Use different colors for different teeth
    #     colors = plt.cm.rainbow(np.linspace(0, 1, len(output.detections)))

    #     # Draw each detection
    #     for i, detection in enumerate(output.detections):
    #         bbox = detection['bbox']  # [xmin, ymin, xmax, ymax]
    #         tooth_id = detection['tooth_id']
    #         score = detection['score']

    #         # Extract bounding box coordinates
    #         x_min, y_min, x_max, y_max = bbox

    #         # Draw the bounding box
    #         rect = patches.Rectangle(
    #             (x_min, y_min), x_max - x_min, y_max - y_min,
    #             linewidth=2, edgecolor=colors[i % len(colors)], facecolor='none'
    #         )
    #         ax.add_patch(rect)

    #         # Add a label with the tooth ID and confidence score
    #         label = f"{tooth_id} ({score:.2f})"
    #         ax.text(
    #             x_min, y_min - 5, label,
    #             fontsize=9, color='white',
    #             bbox=dict(facecolor=colors[i % len(colors)], alpha=0.5)
    #         )

    #     # Save the visualization
    #     save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     plt.tight_layout()
    #     plt.savefig(save_path, dpi=200, bbox_inches='tight')
    #     plt.close()

    #     print(f"Visualization saved to: {save_path}")
    #     return str(save_path)

# class PanoramicXRayToothDetectionTool(BaseTool):
#     """Tool for performing detailed tooth detection analysis of panoramic X-ray images."""

#     name: str = "panoramic_xray_tooth_detection"
#     description: str = (
#         "Detects teeth in panoramic X-ray images and identifies their IDs based on the FDI World Dental Federation notation. "
#         "FDI standard IDs represent teeth as follows: "
#         "11-18 (upper right), 21-28 (upper left), 31-38 (lower left), 41-48 (lower right). "
#         "Returns detection visualization and a list of detected teeth with their bounding boxes and confidence scores. "
#         "Ensure the input image is of high quality for accurate detection."
#     )

#     args_schema: Type[BaseModel] = PanoramicXRayToothDetectionInput

#     model: Any = None
#     device: Optional[str] = "cuda"
#     transform: Any = None
#     temp_dir: Path = Path("temp")
#     organ_map: Dict[str, int] = None

#     def __init__(self, device: Optional[str] = "cuda", temp_dir: Optional[Path] = Path("temp")):
#         """Initialize the segmentation tool with model and temporary directory."""
#         super().__init__()
#         self.model = xrv.baseline_models.chestx_det.PSPNet()
#         self.device = torch.device(device) if device else "cuda"
#         self.model = self.model.to(self.device)
#         self.model.eval()

#         self.transform = torchvision.transforms.Compose(
#             [xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(512)]
#         )

#         self.temp_dir = temp_dir if isinstance(temp_dir, Path) else Path(temp_dir)
#         self.temp_dir.mkdir(exist_ok=True)

#         # Map friendly names to model target indices
#         self.organ_map = {
#             "Left Clavicle": 0,
#             "Right Clavicle": 1,
#             "Left Scapula": 2,
#             "Right Scapula": 3,
#             "Left Lung": 4,
#             "Right Lung": 5,
#             "Left Hilus Pulmonis": 6,
#             "Right Hilus Pulmonis": 7,
#             "Heart": 8,
#             "Aorta": 9,
#             "Facies Diaphragmatica": 10,
#             "Mediastinum": 11,
#             "Weasand": 12,
#             "Spine": 13,
#         }

#     def _align_mask_to_original(
#         self, mask: np.ndarray, original_shape: Tuple[int, int]
#     ) -> np.ndarray:
#         """
#         Align a mask from the transformed (cropped/resized) space back to the full original image.
#         Assumes that the transform does a center crop to a square of side = min(original height, width)
#         and then resizes to (512,512).
#         """
#         orig_h, orig_w = original_shape
#         crop_size = min(orig_h, orig_w)
#         crop_top = (orig_h - crop_size) // 2
#         crop_left = (orig_w - crop_size) // 2

#         # Resize mask (from 512x512) to the cropped region size
#         resized_mask = skimage.transform.resize(
#             mask, (crop_size, crop_size), order=0, preserve_range=True, anti_aliasing=False
#         )
#         full_mask = np.zeros(original_shape)
#         full_mask[crop_top : crop_top + crop_size, crop_left : crop_left + crop_size] = resized_mask
#         return full_mask

#     def _compute_organ_metrics(
#         self, mask: np.ndarray, original_img: np.ndarray, confidence: float
#     ) -> Optional[OrganMetrics]:
#         """Compute comprehensive metrics for a single organ mask."""
#         # Align mask to the original image coordinates if needed
#         if mask.shape != original_img.shape:
#             mask = self._align_mask_to_original(mask, original_img.shape)

#         props = skimage.measure.regionprops(mask.astype(int))
#         if not props:
#             return None

#         props = props[0]
#         area_cm2 = mask.sum() * (self.pixel_spacing_mm / 10) ** 2

#         img_height, img_width = mask.shape
#         cy, cx = props.centroid
#         relative_pos = {
#             "top": cy / img_height,
#             "left": cx / img_width,
#             "center_dist": np.sqrt(((cy / img_height - 0.5) ** 2 + (cx / img_width - 0.5) ** 2)),
#         }

#         organ_pixels = original_img[mask > 0]
#         mean_intensity = organ_pixels.mean() if len(organ_pixels) > 0 else 0
#         std_intensity = organ_pixels.std() if len(organ_pixels) > 0 else 0

#         return OrganMetrics(
#             area_pixels=int(mask.sum()),
#             area_cm2=float(area_cm2),
#             centroid=(float(cy), float(cx)),
#             bbox=tuple(map(int, props.bbox)),
#             width=int(props.bbox[3] - props.bbox[1]),
#             height=int(props.bbox[2] - props.bbox[0]),
#             aspect_ratio=float(
#                 (props.bbox[2] - props.bbox[0]) / max(1, props.bbox[3] - props.bbox[1])
#             ),
#             relative_position=relative_pos,
#             mean_intensity=float(mean_intensity),
#             std_intensity=float(std_intensity),
#             confidence_score=float(confidence),
#         )

#     def _save_visualization(
#         self, original_img: np.ndarray, pred_masks: torch.Tensor, organ_indices: List[int]
#     ) -> str:
#         """Save visualization of original image with segmentation masks overlaid."""
#         plt.figure(figsize=(10, 10))
#         plt.imshow(
#             original_img, cmap="gray", extent=[0, original_img.shape[1], original_img.shape[0], 0]
#         )

#         # Generate color palette for organs
#         colors = plt.cm.rainbow(np.linspace(0, 1, len(organ_indices)))

#         # Process and overlay each organ mask
#         for idx, (organ_idx, color) in enumerate(zip(organ_indices, colors)):
#             mask = pred_masks[0, organ_idx].cpu().numpy()
#             if mask.sum() > 0:
#                 # Align the mask to the original image coordinates
#                 if mask.shape != original_img.shape:
#                     mask = self._align_mask_to_original(mask, original_img.shape)

#                 # Create a colored overlay with transparency
#                 colored_mask = np.zeros((*original_img.shape, 4))
#                 colored_mask[mask > 0] = (*color[:3], 0.3)
#                 plt.imshow(
#                     colored_mask, extent=[0, original_img.shape[1], original_img.shape[0], 0]
#                 )

#                 # Add legend entry for the organ
#                 organ_name = list(self.organ_map.keys())[
#                     list(self.organ_map.values()).index(organ_idx)
#                 ]
#                 plt.plot([], [], color=color, label=organ_name, linewidth=3)

#         plt.title("Segmentation Overlay")
#         plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
#         plt.axis("off")

#         save_path = self.temp_dir / f"segmentation_{uuid.uuid4().hex[:8]}.png"
#         plt.savefig(save_path, bbox_inches="tight", dpi=300)
#         plt.close()

#         return str(save_path)

#     def _run(
#         self,
#         image_path: str,
#         organs: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForToolRun] = None,
#     ) -> Tuple[Dict[str, Any], Dict]:
#         """Run segmentation analysis for specified organs."""
#         try:
#             # Validate and get organ indices
#             if organs:
#                 organs = [o.strip() for o in organs]
#                 invalid_organs = [o for o in organs if o not in self.organ_map]
#                 if invalid_organs:
#                     raise ValueError(f"Invalid organs specified: {invalid_organs}")
#                 organ_indices = [self.organ_map[o] for o in organs]
#             else:
#                 organ_indices = list(self.organ_map.values())
#                 organs = list(self.organ_map.keys())

#             # Load and process image
#             original_img = skimage.io.imread(image_path)
#             if len(original_img.shape) > 2:
#                 original_img = original_img[:, :, 0]

#             img = xrv.datasets.normalize(original_img, 255)
#             img = img[None, ...]
#             img = self.transform(img)
#             img = torch.from_numpy(img)
#             img = img.to(self.device)

#             # Generate predictions
#             with torch.no_grad():
#                 pred = self.model(img)
#             pred_probs = torch.sigmoid(pred)
#             pred_masks = (pred_probs > 0.5).float()

#             # Save visualization
#             viz_path = self._save_visualization(original_img, pred_masks, organ_indices)

#             # Compute metrics for selected organs
#             results = {}
#             for idx, organ_name in zip(organ_indices, organs):
#                 mask = pred_masks[0, idx].cpu().numpy()
#                 if mask.sum() > 0:
#                     metrics = self._compute_organ_metrics(
#                         mask, original_img, float(pred_probs[0, idx].mean().cpu())
#                     )
#                     if metrics:
#                         results[organ_name] = metrics

#             output = {
#                 "segmentation_image_path": viz_path,
#                 "metrics": {organ: metrics.dict() for organ, metrics in results.items()},
#             }

#             metadata = {
#                 "image_path": image_path,
#                 "segmentation_image_path": viz_path,
#                 "original_size": original_img.shape,
#                 "model_size": tuple(img.shape[-2:]),
#                 "pixel_spacing_mm": self.pixel_spacing_mm,
#                 "requested_organs": organs,
#                 "processed_organs": list(results.keys()),
#                 "analysis_status": "completed",
#             }

#             return output, metadata

#         except Exception as e:
#             error_output = {"error": str(e)}
#             error_metadata = {
#                 "image_path": image_path,
#                 "analysis_status": "failed",
#                 "error_traceback": traceback.format_exc(),
#             }
#             return error_output, error_metadata

#     async def _arun(
#         self,
#         image_path: str,
#         organs: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
#     ) -> Tuple[Dict[str, Any], Dict]:
#         """Async version of _run."""
#         return self._run(image_path, organs)
