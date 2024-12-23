from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np


# Custom Dataset Class with Albumentations
class CIFAR10Dataset(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super(CIFAR10Dataset, self).__init__(
            root=root, train=train, download=True, transform=None
        )
        self.albumentations_transform = transform

    def __getitem__(self, index):
        image, label = super(CIFAR10Dataset, self).__getitem__(index)
        image = np.array(image)  # Convert PIL Image to numpy array

        if self.albumentations_transform:
            transformed = self.albumentations_transform(image=image)
            image = transformed["image"]

        return image, label


# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
