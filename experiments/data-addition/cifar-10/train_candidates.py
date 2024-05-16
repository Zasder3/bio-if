import argparse
import os

import torch
from kronfluence.analyzer import Analyzer, prepare_model

from bio_if.tasks.classification import ClassificationTask
from cifar10 import get_dataset, get_model, CIFAR10Loss
from train import train_epoch

USE_CUDA = torch.cuda.is_available()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor-strategy", type=str, default="ekfac")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    return args

def train_model(dataset: torch.utils.data.Dataset, args: argparse.Namespace, k: int):
    """
    Train a model on the given dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to train on.
        args (argparse.Namespace): The arguments for training.
        k (int): The number of indices to train on.

    Returns:
        torch.nn.Module: The trained model.
    """
    model = get_model()
    if USE_CUDA:
        model = model.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), 3e-4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loss = CIFAR10Loss()
    for _ in range(args.epochs):
        epoch_loss = train_epoch(model, optimizer, loss, dataloader)
        print(f"Epoch loss: {epoch_loss}")
    
    if os.path.exists(f"models/{args.factor_strategy}") is False:
        os.makedirs(f"models/{args.factor_strategy}")
    torch.save(model.state_dict(), f"models/{args.factor_strategy}/cifar10_pretrained_seed={args.seed}_k={k}.pth")


def main():
    args = parse_args()

    # Load the dataset
    train_dataset = get_dataset("train")
    candidate_dataset = get_dataset("candidates")

    # Load the model
    task = ClassificationTask()
    model = get_model()
    model = prepare_model(model, task)

    # create analyzer
    analyzer = Analyzer(
        analysis_name=f"cifar10_seed={args.seed}",
        model=model,
        task=task,
    )
    if args.factor_strategy == "random":
        scores = torch.rand(len(candidate_dataset), generator=torch.Generator().manual_seed(args.seed))
    else:
        scores = analyzer.load_pairwise_scores(args.factor_strategy)["all_modules"] # shape V x C
        scores = scores.sum(dim=0) # shape C
    
    
    # grab indices of lowest k scores
    pct_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    k_values = [int(pct * len(scores)) for pct in pct_values]
    score_sorted_indices = scores.argsort()
    indices = [score_sorted_indices[:k] for k in k_values]

    # for every set of indices, train a model and save it
    for k, idx in zip(k_values, indices):
        subset_candidate_dataset = torch.utils.data.Subset(candidate_dataset, idx)
        concat_train_dataset = torch.utils.data.ConcatDataset([train_dataset, subset_candidate_dataset])
        train_model(concat_train_dataset, args, k)

if __name__ == "__main__":
    main()