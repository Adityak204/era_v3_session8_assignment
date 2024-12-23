import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.optim.lr_scheduler import ReduceLROnPlateau
from albumentations.pytorch import ToTensorV2
import numpy as np
from src.classifier import CIFAR_CNN
from src.utils import train, test, load_cifar10_with_albumentations

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    config_dict = {"seed": 42, "batch_size": 64}

    albumentations_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.05),
            A.CoarseDropout(p=0.1, fill=(0.49139968, 0.48215827, 0.44653124)),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2(),
        ]
    )
    train_loader, test_loader = load_cifar10_with_albumentations(
        device=device,
        use_cuda=use_cuda,
        config_dict=config_dict,
        albumentations_transform=albumentations_transform,
    )

    model = CIFAR_CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-6,
    )
    for epoch in range(1, 61):
        print(f"********* Epoch = {epoch} *********")
        train(model, device, train_loader, optimizer, epoch)
        _, acc = test(model, device, test_loader)
        scheduler.step(acc)
        print("LR = ", scheduler.get_last_lr())
