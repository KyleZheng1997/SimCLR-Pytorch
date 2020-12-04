import torch.nn as nn
import torch
from network.resnet import *
from network.head import *


class BackBone(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = resnet50()
        self.head = MLP(2048, dim)
    
    def forward(self, x):
        feat = self.net(x)
        embedding = self.head(feat)
        return embedding

