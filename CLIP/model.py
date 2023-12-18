import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import os
import clip
    
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

class ClipClassify(nn.Module):
    def __init__(self, model): #passed in model
        super().__init__()
        self.vit = model.visual
        self.classification_head = nn.Linear(512, 101, dtype=torch.float32)

    def forward(self, x):
        x = self.vit(x)
        x = x.float()
        x = self.classification_head(x)
        x = F.softmax(x, dim=1)
        return x

class LinearProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.classification_head = nn.Linear(512, 101, dtype=torch.float32)


    def forward(self, x):
        x = x.float()
        x = self.classification_head(x)
        x = F.softmax(x, dim=0)
        return x