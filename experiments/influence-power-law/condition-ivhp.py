import os
from pathlib import Path

import click
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

from kronfluence import Analyzer, prepare_model
from kronfluence.utils.dataset import DataLoaderKwargs
from transformers import AutoTokenizer, AutoModelForMaskedLM

from bio_if.tasks import ESMMLMTask

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

@click.command()
@click.option('--model_name', default="facebook/esm2_t33_650M_UR50D", help='Model name', type=str)
@click.option('--per_device_batch_size', default=16, help='Per device batch size', type=int)
def main(model_name: str, per_device_batch_size: int):
    is_ddp = os.environ.get("WORLD_SIZE") is not None
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Local rank: {local_rank}")
        torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, device_map=f'cuda:{local_rank}')
    tokenizer.model_max_length = float('inf')
    # remove model.esm.contact_head from model
    del model.esm.contact_head

    # process data
    df = pd.read_csv('uniref90_random_10k.csv')
    seqs = df['seq'].tolist()

    ds = Dataset.from_dict({'seq': seqs})
    ds = ds.map(lambda x: tokenizer(x['seq']))

    collator_fn = DataCollatorWithPadding(tokenizer)
    task = ESMMLMTask(tokenizer.all_special_ids, num_layers=len(model.esm.encoder.layer))
    model = prepare_model(model, task)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name=f"{model_name}",
        task=task,
        model=model,
        cpu=False,
    )


    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Beginning Analysis")
    analyzer.fit_all_factors(
        factors_name=f"{model_name}_random_10k",
        dataset=ds.select_columns(["input_ids", "attention_mask"]),
        per_device_batch_size=per_device_batch_size,
        overwrite_output_dir=False,
    )


if __name__ == '__main__':
    main()