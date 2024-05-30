# model.py
import torch
from torch import nn
import torchvision.models as models

class CassavaResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Batch Norm 2d
        self.bn = nn.BatchNorm2d(3)
        
        # Load ResNet-50 and remove the last fc layer
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])

        # Global Average Pooling 2D
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Custom fully connected layers
        self.fc1 = nn.Linear(2048, 8)  # 2048 is the output feature size of ResNet-50
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 5)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Forward pass through the layers
        x = self.bn(x)
        x = self.resnet50(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
