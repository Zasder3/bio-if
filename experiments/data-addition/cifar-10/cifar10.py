from typing import Tuple

import torch
import torch.nn as nn
from torchvision import datasets, transforms

USE_CUDA = torch.cuda.is_available()

def get_dataset(split: str, seed: int = 42) -> datasets.CIFAR10:
    """
    Get the CIFAR10 dataset split and preprocess it by normalizing the pixel values.

    Args:
        split (str): The dataset split to get. One of "train", "candidates", "validation", "test".
        seed (int): The random seed to use for the split.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The dataset split.
    """
    possible_splits = ["train", "candidates", "validation", "test"]
    assert split in possible_splits, f"Split must be one of {possible_splits}, got {split}."
    if split in ["train", "candidates"]:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    if split == "test":
        return datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
    # shuffle the training data using the seed and take the corresponding 10% split
    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    num_train = len(train_dataset)
    indices = torch.randperm(num_train, generator=torch.Generator().manual_seed(seed))
    num_candidates = num_train // 10
    split_idx = possible_splits.index(split)
    dataset = torch.utils.data.Subset(
        train_dataset,
        indices[num_candidates * split_idx: num_candidates * (split_idx + 1)]
    )

    return dataset

class Mul(nn.Module):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)


class Residual(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.module(x)


def construct_resnet9() -> nn.Module:
    # ResNet-9 architecture from https://github.com/MadryLab/trak/blob/main/examples/cifar_quickstart.ipynb.
    def conv_bn(
        channels_in: int,
        channels_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ) -> nn.Module:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                channels_in,
                channels_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            torch.nn.BatchNorm2d(channels_out),
            torch.nn.ReLU(),
        )

    model = torch.nn.Sequential(
        conv_bn(3, 64, kernel_size=3, stride=1, padding=1),
        conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
        Residual(torch.nn.Sequential(conv_bn(128, 128), conv_bn(128, 128))),
        conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.MaxPool2d(2),
        Residual(torch.nn.Sequential(conv_bn(256, 256), conv_bn(256, 256))),
        conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
        torch.nn.AdaptiveMaxPool2d((1, 1)),
        Flatten(),
        torch.nn.Linear(128, 10, bias=False),
        Mul(0.2),
    )
    return model

def get_model() -> torch.nn.Module:
    """
    Get the CIFAR10 model.

    Returns:
        torch.nn.Module: The CIFAR10 model.
    """
    # create a ResNet-9 model
    # model = construct_resnet9()
    # return model
    dim = 2048
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(1024*3, dim),
        torch.nn.GELU(),
        torch.nn.Linear(dim, dim),
        torch.nn.GELU(),
        torch.nn.Linear(dim, 10),
    )


class CIFAR10Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.reset_accuracy()

    def reset_accuracy(self):
        self.total_correct = 0
        self.total_samples = 0
    
    def get_accuracy(self):
        return self.total_correct / self.total_samples
    
    def forward(self, model: torch.nn.Module, batch: Tuple[torch.Tensor, torch.Tensor], cache_accuracy: bool = False) -> torch.Tensor:
        inputs, labels = batch
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        logits = model(inputs)
        if cache_accuracy:
            self.total_correct += (logits.argmax(1) == labels).float().sum()
            self.total_samples += len(labels)
        return torch.nn.functional.cross_entropy(logits, labels)
    