import os

import click
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

from kronfluence import Analyzer, prepare_model
from kronfluence.utils.dataset import DataLoaderKwargs

from bio_if.tasks import ESMMLMTask

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@click.command()
@click.option(
    "--model_name", default="facebook/esm2_t33_650M_UR50D", help="Model name", type=str
)
@click.option(
    "--per_device_batch_size", default=8, help="Per device batch size", type=int
)
@click.option(
    "--training_data", default="uniref50_random_10k.csv", help="Training data", type=str
)
@click.option(
    "--sequence_of_interest", help="Sequence of interest in a FASTA", type=str
)
@click.option("--compile", default=False, help="Compile model", type=bool)
def main(
    model_name: str,
    per_device_batch_size: int,
    training_data: str,
    sequence_of_interest: str,
    compile: bool,
):
    is_ddp = os.environ.get("WORLD_SIZE") is not None
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Local rank: {local_rank}")
        torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(
        model_name, device_map=f"cuda:{local_rank}"
    )
    tokenizer.model_max_length = float("inf")
    # remove model.esm.contact_head from model
    del model.esm.contact_head

    # process data
    if training_data.endswith(".csv"):
        df = pd.read_csv(training_data)
        seqs = df["seq"].tolist()
        # sort by length in descending order
        seqs = sorted(seqs, key=len, reverse=True)

        ds = Dataset.from_dict({"seq": seqs})
        ds = ds.map(lambda x: tokenizer(x["seq"]))

    # process sequence of interest FASTA
    with open(sequence_of_interest, "r") as f:
        seq = f.readlines()[1:]
        seq = [s.strip() for s in seq]
        seq = "".join(seq)

    test_ds = Dataset.from_dict({"seq": [seq]})
    test_ds = test_ds.map(lambda x: tokenizer(x["seq"]))

    collator_fn = DataCollatorWithPadding(tokenizer)
    task = ESMMLMTask(
        tokenizer.all_special_ids, num_layers=len(model.esm.encoder.layer)
    )
    model = prepare_model(model, task)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    if compile:
        model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name=f"{model_name}",
        task=task,
        model=model,
        cpu=False,
    )

    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Loading Factors")
    training_data_name = training_data.split("/")[-1].split(".")[0]
    sequence_of_interest_name = sequence_of_interest.split("/")[-1].split(".")[0]
    analyzer.compute_pairwise_scores(
        scores_name=f"{training_data_name}_{sequence_of_interest_name}",
        factors_name=f"{model_name}_random_10k",
        overwrite_output_dir=False,
        train_dataset=ds.select_columns(["input_ids", "attention_mask"]),
        query_dataset=test_ds.select_columns(["input_ids", "attention_mask"]),
        per_device_query_batch_size=1,
        per_device_train_batch_size=per_device_batch_size,
    )


if __name__ == "__main__":
    main()
