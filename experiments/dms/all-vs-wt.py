import os

import click

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch
import pandas as pd

from kronfluence import Analyzer, prepare_model
from kronfluence.utils.dataset import DataLoaderKwargs

from bio_if.tasks import ESMMLMTask

# model_name = "facebook/esm2_t36_3B_UR50D"
# model_name = "facebook/esm2_t33_650M_UR50D"
# model_name = "facebook/esm2_t30_150M_UR50D"
# model_name = "facebook/esm2_t12_35M_UR50D"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@click.command()
@click.option(
    "--model_name", default="facebook/esm2_t12_35M_UR50D", help="Model name", type=str
)
@click.option("--seed", default=0, help="Seed", type=int)
@click.option(
    "--per_device_batch_size", default=128, help="Per device batch size", type=int
)
@click.option(
    "--study_name", default="GCN4_YEAST_Staller_2018", help="Study name", type=str
)
def main(model_name: str, seed: int, per_device_batch_size: int, study_name: str):
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

    all_dms_df = pd.read_csv("DMS_substitutions.csv")
    seq = all_dms_df[all_dms_df["DMS_id"] == study_name]["target_seq"].iloc[0]
    seqs_df = pd.read_csv(f"studies/{study_name}.csv")
    # sample 1000 sequences without replacement
    seqs_df = seqs_df.sample(n=1000, random_state=seed, replace=False)
    seqs = seqs_df["mutated_sequence"].tolist()

    ds = Dataset.from_dict({"seq": seqs})
    ds = ds.map(lambda x: tokenizer(x["seq"]))

    test_ds = Dataset.from_dict({"seq": [seq]})
    test_ds = test_ds.map(lambda x: tokenizer(x["seq"]))

    collator_fn = DataCollatorWithPadding(tokenizer)
    task = ESMMLMTask(
        tokenizer.all_special_ids, num_layers=len(model.esm.encoder.layer)
    )
    model = prepare_model(model, task)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)
    model = torch.compile(model)

    analyzer = Analyzer(
        analysis_name=f"{model_name}_{study_name}_seed={seed}",
        task=task,
        model=model,
        cpu=False,
    )

    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Beginning Analysis")
    analyzer.fit_all_factors(
        factors_name=f"{model_name}_{study_name}",
        dataset=ds.select_columns(["input_ids", "attention_mask"]),
        per_device_batch_size=per_device_batch_size,
        overwrite_output_dir=False,
    )

    analyzer.compute_pairwise_scores(
        scores_name=f"{model_name}_{study_name}",
        factors_name=f"{model_name}_{study_name}",
        overwrite_output_dir=False,
        train_dataset=ds.select_columns(["input_ids", "attention_mask"]),
        query_dataset=test_ds.select_columns(["input_ids", "attention_mask"]),
        per_device_query_batch_size=1,
        per_device_train_batch_size=per_device_batch_size,
    )

    # recursively delete the factors_facebook folder
    if local_rank == 0:
        path = f"analyses/{model_name}_{study_name}_seed={seed}/factors_facebook"
        assert os.path.exists(path)
        if os.path.exists(path):
            # Walk through the directory
            for root, dirs, files in os.walk(path, topdown=False):
                # Delete all files
                for file in files:
                    os.remove(os.path.join(root, file))
                # Delete all directories
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            # Delete the main directory
            os.rmdir(path)
            print(f"Successfully deleted the folder: {path}")
        else:
            print(f"The folder does not exist: {path}")


if __name__ == "__main__":
    main()
