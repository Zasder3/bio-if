from pathlib import Path
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding
from datasets import Dataset
import torch

from kronfluence import Analyzer, prepare_model
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.state import release_memory
from transformers import AutoTokenizer, AutoModelForMaskedLM

from bio_if.tasks import MLMTask

MODEL_NAME = "facebook/esm2_t36_3B_UR50D"
# MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
# MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
# MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SEED = 0

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, device_map='cuda')
    tokenizer.model_max_length = float('inf')
    # remove model.esm.contact_head from model
    del model.esm.contact_head
    model = torch.compile(model)

    seqs_df = pd.read_csv('../../src/bio_if/data/plmbias/len_rev_common_proteins_progen_esm_loglikelihood.csv')
    seqs_df = seqs_df[seqs_df['Length'] <= 510] # 510 to make the tokenized seqs 512
    seqs = seqs_df['sequence'].tolist()

    # sample a single sequence
    np.random.seed(SEED)
    seq = np.random.choice(seqs)

    # make 128 copies of the sequence
    seqs = [list(seq) for _ in range(128)]
    # make up to 16 random mutations in each sequence
    for i in range(128):
        for _ in range(np.random.randint(1, 16)):
            idx = np.random.randint(0, len(seq))
            seqs[i][idx] = np.random.choice(list(AMINO_ACIDS))
    seqs = ["".join(seq) for seq in seqs]
    
    # save sequences to a file
    df = pd.DataFrame({
        'sequence': [seq] + seqs,
        'mutations': [0] + [sum([a != b for a, b in zip(seq, seqs[i])]) for i in range(len(seqs))]
    })
    if not Path(f"analyses/{MODEL_NAME}_self_self_seed={SEED}").exists():
        Path(f"analyses/{MODEL_NAME}_self_self_seed={SEED}").mkdir(parents=True)
    df.to_csv(f"analyses/{MODEL_NAME}_self_self_seed={SEED}/seqs.csv", index=False)

    ds = Dataset.from_dict({'seq': seqs})
    ds = ds.map(lambda x: tokenizer(x['seq']))

    test_ds = Dataset.from_dict({'seq': [seq]})
    test_ds = test_ds.map(lambda x: tokenizer(x['seq']))


    collator_fn = DataCollatorWithPadding(tokenizer)
    task = MLMTask(tokenizer.all_special_ids)
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=f"{MODEL_NAME}_self_self_seed={SEED}",
        task=task,
        model=model,
        cpu=False,
    )


    dataloader_kwargs = DataLoaderKwargs(collate_fn=collator_fn)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    print("Beginning Analysis")
    analyzer.fit_all_factors(
        factors_name=f"{MODEL_NAME}_self_self",
        dataset=ds.select_columns(["input_ids", "attention_mask"]),
        per_device_batch_size=None,
        overwrite_output_dir=False,
        initial_per_device_batch_size_attempt=128,
    )

    analyzer.compute_pairwise_scores(
        scores_name=f"{MODEL_NAME}_self_self",
        factors_name=f"{MODEL_NAME}_self_self",
        overwrite_output_dir=True,
        query_dataset=ds.select_columns(["input_ids", "attention_mask"]),
        train_dataset=test_ds.select_columns(["input_ids", "attention_mask"]),
        per_device_query_batch_size=1,
    )


if __name__ == '__main__':
    main()