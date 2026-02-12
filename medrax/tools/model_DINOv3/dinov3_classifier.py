import torch
import torch.nn as nn
from transformers import AutoModel
from safetensors.torch import load_file

class DinoV3Classifier(nn.Module):
    """Complete DinoV3 classifier model."""
    
    def __init__(self, task_name, num_classes, hidden_size=1024, dropout=0.4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov3-convnext-base-pretrain-lvd1689m")
        if "intraoral_abnormal_9class" in task_name or "oscc_5class" in task_name:
            self.norm = nn.BatchNorm1d(hidden_size, affine=False)
        elif "leukoplakia_3class" in task_name or "oscc_multi_6class" in task_name:
            self.norm = nn.BatchNorm1d(hidden_size, affine=True)
        else:
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, pixel_values):
        # Extract features from backbone
        outputs = self.backbone(pixel_values)
        
        last_hidden = outputs.pooler_output
        last_hidden = self.norm(last_hidden)
        last_hidden = self.dropout(last_hidden)
        logits = self.classifier(last_hidden)
        return logits