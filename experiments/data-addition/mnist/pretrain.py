import os

import torch

from train import train_epoch
from mnist import get_dataset, get_model, MNISTLoss

RANDOM_SEED = 42
USE_CUDA = torch.cuda.is_available()
EPOCHS = 20
BATCH_SIZE = 128


def pretrain_model(seed: int) -> torch.nn.Module:
    """
    Pretrain a model on the MNIST dataset.

    Args:
        seed (int): The random seed to use for the split.

    Returns:
        torch.nn.Module: The pretrained model.
    """
    train_dataset = get_dataset("train", seed)
    model = get_model()
    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    loss = MNISTLoss()
    for _ in range(EPOCHS):
        epoch_loss = train_epoch(model, optimizer, loss, dataloader)
        print(f"Epoch loss: {epoch_loss}")
    return model

def main():
    model = pretrain_model(RANDOM_SEED)
    if os.path.exists("models") is False:
        os.makedirs("models")
    torch.save(model.state_dict(), f"models/mnist_pretrained_seed={RANDOM_SEED}.pth")

if __name__ == "__main__":
    main()
