from typing import Tuple

import torch
from torchvision import datasets, transforms

USE_CUDA = torch.cuda.is_available()

def get_dataset(split: str, seed: int = 42) -> datasets.MNIST:
    """
    Get the MNIST dataset split and preprocess it by normalizing the pixel values.

    Args:
        split (str): The dataset split to get. One of "train", "candidates", "validation", "test".
        seed (int): The random seed to use for the split.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The dataset split.
    """
    possible_splits = ["train", "candidates", "validation", "test"]
    assert split in possible_splits, f"Split must be one of {possible_splits}, got {split}."
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if split == "test":
        return datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform,
        )
    # shuffle the training data using the seed and take the corresponding 10% split
    train_dataset = datasets.MNIST(
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

def get_model() -> torch.nn.Module:
    """
    Get the MNIST model.

    Returns:
        torch.nn.Module: The MNIST model.
    """
    return torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(784, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 1024),
        torch.nn.GELU(),
        torch.nn.Linear(1024, 10),
    )

class MNISTLoss(torch.nn.Module):
    def forward(self, model: torch.nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, labels = batch
        if USE_CUDA:
            inputs = inputs.cuda()
            labels = labels.cuda()
        logits = model(inputs)
        return torch.nn.functional.cross_entropy(logits, labels)