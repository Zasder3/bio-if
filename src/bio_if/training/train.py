import logging
from bio_if.evaluation.fitness import evaluate_zero_shot_fitness_prediction
import wandb
import torch
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


def validate_loader(model, loader, scaler, desc):
    valid_loss = 0
    n_tokens_total = 0
    for batch in tqdm(loader, desc=desc):
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(**batch)
        n_tokens = torch.sum(batch["labels"] != -100)
        n_tokens_total += n_tokens
        valid_loss += output["loss"] * n_tokens
    return valid_loss / n_tokens_total


def train_one_epoch(
    model: nn.Module,
    tokenizer: torch.nn.Module,
    iid_train_loader: torch.utils.data.DataLoader,
    iid_val_loader: torch.utils.data.DataLoader,
    ood_val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    fitness_dataset_path: str,
    epoch: int,
    scaler: torch.cuda.amp.GradScaler,
):
    logger.info(f"Beginning Epoch {epoch}")
    model.train()
    for batch in tqdm(iid_train_loader, desc="Training"):
        optimizer.zero_grad()
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            output = model(**batch)
        scaler.scale(output["loss"]).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log to wandb
        wandb.log(
            {
                "train_loss": output["loss"].item(),
                "epoch": epoch,
                "step": optimizer.state_dict()["state"][0]["step"],
            }
        )
    model.eval()

    with torch.no_grad():
        iid_valid_loss = validate_loader(
            model, iid_val_loader, scaler, "Finetuning Validation"
        )
        ood_valid_loss = validate_loader(
            model, ood_val_loader, scaler, "Original Distribution Validation"
        )

    zero_shot_fitness_spearman = evaluate_zero_shot_fitness_prediction(
        model, tokenizer, fitness_dataset_path
    )
    # Log validation metrics to wandb
    wandb.log(
        {
            "finetuning_valid_loss": iid_valid_loss.item(),
            "original_distribution_valid_loss": ood_valid_loss.item(),
            "zero_shot_fitness_spearman": zero_shot_fitness_spearman,
            "epoch": epoch,
        }
    )

    model.train()
