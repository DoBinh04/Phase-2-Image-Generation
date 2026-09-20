"""
MNIST data loading.

Images are padded from 28x28 to 32x32 (so the U-Net's two 2x downsamples
land on clean integer resolutions: 32 -> 16 -> 8) and scaled to [-1, 1],
which is the standard range diffusion models are trained on.

To use CIFAR-10 instead, swap `datasets.MNIST` for `datasets.CIFAR10` and
set in_channels=3 in the model — CIFAR-10 is already 32x32 so the Pad(2)
transform can be dropped.
"""
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_mnist_dataloader(batch_size=128, root="./data", train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.Pad(2),                       # 28x28 -> 32x32
        transforms.ToTensor(),                   # [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1),  # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                         num_workers=num_workers, drop_last=True, pin_memory=True)
    return loader


def get_cifar10_dataloader(batch_size=128, root="./data", train=True, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),                   # [0, 1], already 32x32
        transforms.Lambda(lambda x: x * 2 - 1),  # [0, 1] -> [-1, 1]
    ])
    dataset = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=train,
                         num_workers=num_workers, drop_last=True, pin_memory=True)
    return loader
