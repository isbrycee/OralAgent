import torch
import torch.nn as nn
from transformers import AutoModel
from safetensors.torch import load_file

class DinoV3Classifier(nn.Module):
    """Complete DinoV3 classifier model."""
    
    def __init__(self, task_name, num_classes, hidden_size=1024, dropout=0.4):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("facebook/dinov3-convnext-base-pretrain-lvd1689m")
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_size, affine=True),
            nn.Linear(hidden_size, num_classes))
    
    def forward(self, pixel_values):
        # Extract features from backbone
        outputs = self.backbone(pixel_values)
        
        last_hidden = outputs.pooler_output
        last_hidden = self.head(last_hidden)
        logits = last_hidden
        return logits