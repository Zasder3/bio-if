import logging
import click
import polars as pl
import numpy as np
import torch
import wandb
from transformers import DataCollatorForLanguageModeling

from bio_if.evaluation.fitness import evaluate_zero_shot_fitness_prediction
from bio_if.training.data import MinimalSequenceDataset
from bio_if.training.model import load_esm_model
from bio_if.training.train import train_one_epoch, validate_loader

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--model_name",
    default="facebook/esm2_t33_650M_UR50D",
    help="Name of the ESM model to load",
)
@click.option(
    "--dms_id",
    default="Specify the DMS study ID",
)
@click.option(
    "--train_data_path",
    help="Path to the CSV file containing sequences",
)
@click.option(
    "--ood_val_data_path",
    help="Path to the CSV file containing sequences of original data",
)
@click.option(
    "--fitness_dataset_path",
    help="Path to the directory containing the fitness dataset",
)
@click.option(
    "--seed",
    default=42,
    help="Seed for reproducibility",
)
@click.option(
    "--batch_size",
    default=16,
    help="Batch size for training",
)
@click.option(
    "--lr",
    default=1e-6,
    help="Learning rate for training",
)
@click.option(
    "--max_num_train_sequences",
    default=1000,
)
@click.option(
    "--num_epochs",
    default=5,
    help="Number of epochs to train the model",
)
@click.option(
    "--fp16",
    default=False,
    help="Whether to use mixed precision training",
    is_flag=True,
)
def main(
    model_name: str,
    dms_id: str,
    train_data_path: str,
    ood_val_data_path: str,
    fitness_dataset_path: str,
    seed: int,
    batch_size: int,
    lr: float,
    max_num_train_sequences: int,
    num_epochs: int,
    fp16: bool,
):
    # initialize wandb and random generator
    logging.basicConfig(level=logging.INFO)
    config = {
        "model_name": model_name.split("/")[-1],
        "study": dms_id,
        "seed": seed,
        "batch_size": batch_size,
        "lr": lr,
        "num_epochs": num_epochs,
    }
    wandb.init(
        project="bio-if",
        config=config,
        name=f"{config['model_name']}_{config['study']}_{config['seed']}_full",
    )
    random_generator = np.random.default_rng(seed)

    # load model
    model, tokenizer = load_esm_model(model_name)
    model = model.cuda()
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)

    # load data, we refer to fine-tuning data as iid and the original data as ood
    iid_df = pl.scan_parquet(train_data_path)
    iid_df = iid_df.filter(pl.col("DMS_id") == dms_id)
    iid_df = iid_df.collect()
    # trim sequences using Target Starting Position - 1 to Target Ending Position
    iid_df = iid_df.with_columns(
        pl.col("seqs").str.slice(
            pl.col("Target Starting Position") - 1,
            pl.col("Target Ending Position") - pl.col("Target Starting Position") + 1,
        )
    )
    iid_seqs = iid_df["seqs"].to_list()[:max_num_train_sequences]
    ood_seqs = pl.read_csv(ood_val_data_path)["seqs"].to_list()[:1024]
    ood_seqs = sorted(
        ood_seqs, key=len, reverse=True
    )  # sort by length in descending order to speed up inference
    # split data into train and validation 80/20 using the given seed
    shuffled_seqs = random_generator.permutation(iid_seqs)
    split_idx = int(0.8 * len(shuffled_seqs))
    iid_train_seqs = shuffled_seqs[:split_idx]
    iid_val_seqs = shuffled_seqs[split_idx:]
    iid_train_dataset = MinimalSequenceDataset(tokenizer, iid_train_seqs)
    iid_val_dataset = MinimalSequenceDataset(tokenizer, iid_val_seqs)
    ood_val_dataset = MinimalSequenceDataset(tokenizer, ood_seqs)
    collator_fn = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, return_tensors="pt"
    )
    iid_train_loader = torch.utils.data.DataLoader(
        iid_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator_fn,
        drop_last=True,
    )
    iid_val_loader = torch.utils.data.DataLoader(
        iid_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator_fn
    )
    ood_val_loader = torch.utils.data.DataLoader(
        ood_val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collator_fn
    )

    # benchmark validation and 0-shot fitness prediction before training
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
    wandb.log(
        {
            "finetuning_valid_loss": iid_valid_loss.item(),
            "original_distribution_valid_loss": ood_valid_loss.item(),
            "zero_shot_fitness_spearman": zero_shot_fitness_spearman,
            "epoch": -1,
        }
    )

    # train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(num_epochs):
        train_one_epoch(
            model,
            tokenizer,
            iid_train_loader,
            iid_val_loader,
            ood_val_loader,
            optimizer,
            fitness_dataset_path,
            i,
            scaler,
        )


if __name__ == "__main__":
    main()
