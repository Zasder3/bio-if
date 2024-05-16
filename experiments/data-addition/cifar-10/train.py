from typing import Callable, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    data_loader: torch.utils.data.DataLoader,
    verbose: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    tqdm_data_loader = tqdm(data_loader, disable=not verbose)
    for batch in tqdm_data_loader:
        optimizer.zero_grad()
        loss = loss_fn(model, batch)
        loss.backward()
        optimizer.step()
        tqdm_data_loader.set_postfix(loss=loss.item())
        total_loss += loss.item() * len(batch[0])
    return total_loss / len(data_loader.dataset)

def validate(
    model: nn.Module,
    loss_fn: Callable[[nn.Module, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor],
    data_loader: torch.utils.data.DataLoader,
    verbose: bool = True,
) -> float:
    model.eval()
    total_loss = 0.0
    tqdm_data_loader = tqdm(data_loader, disable=not verbose)
    with torch.no_grad():
        for batch in tqdm_data_loader:
            loss = loss_fn(model, batch, cache_accuracy=True)
            tqdm_data_loader.set_postfix(loss=loss.item())
            total_loss += loss.item() * len(batch[0])
    return total_loss / len(data_loader.dataset)
