import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import matplotlib.pyplot as plt
from torchsummary import summary
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from src.support_model import DepthwiseSeparableConv


# {(inp + 2xP - D(K-1) -1)/s + 1} OR {(inp + 2xp - k)/s + 1}
class CIFAR_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 32,3>32,16|RF:3,J:1
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 32,16>32,32|RF:5,J:1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 32,32>32,64|RF:7,J:1
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )
        self.transition_1 = nn.Conv2d(
            in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False
        )  # 32,64>32,16|RF:7,J:1
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 32,16>32,32|RF:9,J:1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 32,32>32,32|RF:11,J:1
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                dilation=2,
                bias=False,
            ),  # 32,32>16,64|RF:13,J:2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.15)
        )
        self.transition_2 = nn.Conv2d(
            in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False
        )  # 16,64>16,16|RF:13,J:2
        self.block_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 16,16>16,32|RF:17,J:2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 16,32>16,32|RF:21,J:2
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(3, 3),
                padding=1,
                dilation=3,
                bias=False,
            ),  # 16,32>8,64|RF:25,J:4
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Dropout(0.15)
        )
        self.transition_3 = nn.Conv2d(
            in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False
        )  # 8,64>8,16|RF:25,J:4
        self.block_4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 8,16>8,32|RF:33,J:4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=(3, 3),
                padding=1,
                bias=False,
            ),  # 8,32>8,32|RF:41,J:4
            nn.BatchNorm2d(32),
            nn.ReLU(),
            DepthwiseSeparableConv(
                in_channels=32, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 1,64
        self.fc = nn.Linear(128, 10)  # 1,64>10

    def forward(self, x):
        x = self.block_1(x)
        x = self.transition_1(x)
        x = self.block_2(x)
        x = self.transition_2(x)
        x = self.block_3(x)
        x = self.transition_3(x)
        x = self.block_4(x)
        x = self.gap(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
