from typing import Dict, Optional
import time

from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import default_data_collator
from datasets import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from kronfluence import Analyzer, prepare_model
from kronfluence.task import Task
from kronfluence.utils.dataset import DataLoaderKwargs
from kronfluence.utils.state import release_memory

torch.cuda.set_device(7)

BATCH_TYPE = Dict[str, torch.Tensor]
# MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
# MODEL_NAME = "facebook/esm2_t30_150M_UR50D"
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"

class MLMTask(Task):
    def compute_train_loss(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
        sample: bool = False,
    ) -> torch.Tensor:
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        if not sample:
            return (F.cross_entropy(logits.transpose(1,2), batch["input_ids"], reduction='none') * batch["attention_mask"]).sum()
        with torch.no_grad():
            sampled_labels = torch.distributions.Categorical(logits=logits).sample()
        return (F.cross_entropy(logits.transpose(1,2), sampled_labels.detach(), reduction='none') * batch["attention_mask"]).sum()

    def compute_measurement(
        self,
        batch: BATCH_TYPE,
        model: nn.Module,
    ) -> torch.Tensor:
        # Copied from: https://github.com/MadryLab/trak/blob/main/trak/modelout_functions.py.
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

        labels = batch["input_ids"]
        logits_correct = logits.view(-1, logits.size(-1))
        logits_correct = logits_correct[torch.arange(len(labels.view(-1))), labels.view(-1)]
        logits_correct = logits_correct.view(*labels.size())

        cloned_logits = logits.clone()
        cloned_logits = cloned_logits.view(-1, cloned_logits.size(-1))
        cloned_logits[torch.arange(len(labels.view(-1))), labels.view(-1)] = float("-inf")
        cloned_logits = cloned_logits.view(*logits.size())

        margins = logits_correct - cloned_logits.logsumexp(dim=-1)
        margins = margins * batch["attention_mask"]
        return -margins.sum()

    def get_attention_mask(self, batch: BATCH_TYPE) -> Optional[torch.Tensor]:
        return batch["attention_mask"]

def read_fasta(fasta_file):
    with open(fasta_file) as f:
        lines = f.readlines()
    seqs = []
    for line in lines:
        if line[0] == '>':
            if seqs:
                seqs[-1] = ''.join(seqs[-1])
            seqs.append([])
        else:
            seqs[-1].append(line.strip())
    seqs[-1] = ''.join(seqs[-1])
    return seqs

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, device_map='cuda:7')
    # remove model.esm.contact_head from model
    del model.esm.contact_head

    seqs = read_fasta('partial.fasta')

    ds = Dataset.from_dict({'seq': [seq[:256] for seq in seqs]})
    ds = ds.map(lambda x: tokenizer(x['seq'], padding='max_length', truncation=True, max_length=256, return_tensors='pt'), batched=True)

    task = MLMTask()
    model = prepare_model(model, task)

    analyzer = Analyzer(
        analysis_name=f"{MODEL_NAME}_benchmarking",
        task=task,
        model=model,
        cpu=False,
    )
    dataloader_kwargs = DataLoaderKwargs(collate_fn=default_data_collator)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    all_times = []
    for train_sizes in [32, 64, 128, 256, 512, 1024]:
        print(f"Training size: {train_sizes}")
        release_memory()

        # time the following
        start = time.time()
        analyzer.fit_all_factors(
            factors_name=f"{MODEL_NAME}_benchmarking_n={train_sizes}",
            dataset=Dataset.from_dict(ds[:train_sizes]),
            # dataset=ds,
            per_device_batch_size=None,
            overwrite_output_dir=False,
            initial_per_device_batch_size_attempt=1024,
        )

        analyzer.compute_pairwise_scores(
            scores_name=f"{MODEL_NAME}_benchmarking_n={train_sizes}",
            factors_name=f"{MODEL_NAME}_benchmarking_n={train_sizes}",
            overwrite_output_dir=True,
            query_dataset=Dataset.from_dict(ds[:256]),
            train_dataset=Dataset.from_dict(ds[:train_sizes]),
            per_device_query_batch_size=8,
        )
        end = time.time()
        print(f"Time taken for {train_sizes}: {end-start}")
        # save time to file
        all_times.append(end-start)
        torch.save(all_times, f'{MODEL_NAME.split("/")[-1]}_times.pt')


if __name__ == '__main__':
    main()
