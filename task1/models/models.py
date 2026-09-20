import torch
import open_clip
from torchvision import datasets, models, transforms

resnet_50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V2')
vit_model = models.vit_b_16(weights='ViT_B_16_Weights.IMAGENET1K_V1')

