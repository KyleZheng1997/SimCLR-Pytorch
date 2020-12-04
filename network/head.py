from network.resnet import resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, dim_mlp=2048, dim_out=128):
        super().__init__()

        self.linear1 = nn.Linear(dim_mlp, dim_mlp)
        self.bn1 = nn.BatchNorm1d(dim_mlp)
        self.relu = nn.ReLU(True)
        self.linear2 = nn.Linear(dim_mlp, dim_out)
        self.bn2 = nn.BatchNorm1d(dim_out)
        
        
    def forward(self, x):
        x = self.linear1(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn1(x).squeeze(-1).squeeze(-1)
        x = self.relu(x)
        x = self.linear2(x).unsqueeze(-1).unsqueeze(-1)
        x = self.bn2(x).squeeze(-1).squeeze(-1)
        return x


class LinearHead(nn.Module):
    def __init__(self, net, dim_in=2048, dim_out=1000):
        super().__init__()
        self.net = net
        self.fc = nn.Linear(dim_in, dim_out)

        for param in self.net.parameters():
            param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        with torch.no_grad():
            feat = self.net(x)
        return self.fc(feat)

