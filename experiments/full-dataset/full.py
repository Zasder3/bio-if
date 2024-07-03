import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Levenshtein import distance
from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from datasets import Dataset

from kronfluence import Analyzer, FactorArguments, prepare_model
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.state import release_memory
from transformers import AutoTokenizer, AutoModelForMaskedLM

from bio_if.tasks import MLMTask

# MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
# MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
# MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 1

def main():
    is_ddp = os.environ.get("WORLD_SIZE") is not None
    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Local rank: {local_rank}")
        torch.distributed.init_process_group(backend="nccl")
    else:
        local_rank = 0
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, device_map=f'cuda:{local_rank}')
    tokenizer.model_max_length = float('inf')
    # remove model.esm.contact_head from model
    del model.esm.contact_head

    seqs_df = pd.read_csv('../../src/bio_if/data/plmbias/len_rev_common_proteins_progen_esm_loglikelihood.csv')
    seqs_df = seqs_df[seqs_df['Length'] <= 510] # 510 to make the tokenized seqs 512
    seqs = seqs_df['sequence'].tolist()
    
    # save sequences to a file
    df = pd.DataFrame({
        'sequence': seqs,
        'mutations': [distance(seqs[0], seqs[i]) for i in range(len(seqs))]
    })
    if not Path(f"analyses/{MODEL_NAME}_full_seed={SEED}").exists():
        Path(f"analyses/{MODEL_NAME}_full_seed={SEED}").mkdir(parents=True)
    df.to_csv(f"analyses/{MODEL_NAME}_full_seed={SEED}/seqs.csv", index=False)

    ds = Dataset.from_dict({'seq': seqs})
    ds = ds.map(lambda x: tokenizer(x['seq']))

    collator_fn = DataCollatorWithPadding(tokenizer)
    task = MLMTask(tokenizer.all_special_ids)
    model = prepare_model(model, task)
    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model)

    analyzer = Analyzer(
        analysis_name=f"{MODEL_NAME}_full_seed={SEED}",
        task=task,
        model=model,
        cpu=False,
    )


    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator_fn)
    # factor_args = FactorArguments(
    #     activation_covariance_dtype=torch.bfloat16,
    #     gradient_covariance_dtype=torch.bfloat16,
    #     eigendecomposition_dtype=torch.bfloat16,
    #     lambda_dtype=torch.bfloat16,
    #     covariance_module_partition_size=2,
    #     lambda_module_partition_size=2,
    #     lambda_iterative_aggregate=True,
    # )
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Beginning Analysis")
    analyzer.fit_all_factors(
        factors_name=f"{MODEL_NAME}_full",
        dataset=ds.select_columns(["input_ids", "attention_mask"]),
        per_device_batch_size=1,
        overwrite_output_dir=False,
        # factor_args=factor_args,
        # initial_per_device_batch_size_attempt=16,
    )

    analyzer.compute_pairwise_scores(
        scores_name=f"{MODEL_NAME}_full",
        factors_name=f"{MODEL_NAME}_full",
        overwrite_output_dir=True,
        query_dataset=ds.select_columns(["input_ids", "attention_mask"]),
        train_dataset=ds.select_columns(["input_ids", "attention_mask"]),
        per_device_query_batch_size=16,
        per_device_train_batch_size=16,
    )


if __name__ == '__main__':
    main()
