import functools
import os
import torch
import pandas as pd
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

def get_one_shot_log_likelihood(model, tokenizer, seqs, eps=1e-3):
    model.eval()
    device = torch.device(f'cuda:{dist.get_rank()}')
    max_length = max([len(seq) for seq in seqs])
    tokenized = tokenizer(seqs, padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')

    batch = {k: v.to(device) for k, v in tokenized.items()}
    label = batch['input_ids']
    label_special = torch.zeros_like(label, dtype=torch.bool)
    for special_id in tokenizer.all_special_ids:
        label_special |= label == special_id
    # label = label.masked_fill(label_special, -100)

    with (torch.no_grad(), torch.cuda.amp.autocast()):
        # outputs = model(**batch).logits.transpose(1, 2) # [batch_size, num_tokens, seq_len]
        outputs = model(**batch).logits # [batch_size, seq_len, num_tokens]

    # individual_lls = torch.nn.functional.cross_entropy(outputs, label, reduction='none') # [batch_size, seq_len]
    # return -(individual_lls.sum(dim=-1) / (~label_special).sum(dim=-1)).numpy(force=True)
    individual_probs = torch.softmax(outputs, dim=-1).transpose(1, 2) # [batch_size, num_tokens, seq_len]
    individual_probs = individual_probs.gather(dim=1, index=label.unsqueeze(1)).squeeze(1) # [batch_size, seq_len]
    individual_probs = (individual_probs * 2 - 1).clamp(min=eps)
    log_probs = torch.log(individual_probs)
    log_probs = log_probs.masked_fill(label_special, 0)
    return (log_probs.sum(dim=-1) / (~label_special).sum(dim=-1)).numpy(force=True)

def main():
    # initialize torch distributed
    dist.init_process_group()
    df = pd.read_csv('DMS_substitutions.csv')

    seqs = df['target_seq'].tolist()
    all_lls = {}
    model_names = [
        ("facebook/esm2_t48_15B_UR50D", "ESM2 (15B) Log Likelihood"),
        # ("facebook/esm2_t36_3B_UR50D", "ESM2 (3B) Log Likelihood"),
        # ("facebook/esm2_t33_650M_UR50D", "ESM2 (650M) Log Likelihood"),
        # ("facebook/esm2_t30_150M_UR50D", "ESM2 (150M) Log Likelihood"),
        # ("facebook/esm2_t12_35M_UR50D", "ESM2 (35M) Log Likelihood"),
        # ("facebook/esm2_t6_8M_UR50D", "ESM2 (8M) Log Likelihood"),
    ]
    for hf_model_name, col_name in model_names:
        print(col_name)
        all_lls[col_name] = []

        auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=1e6
        )
            
        model = AutoModelForMaskedLM.from_pretrained(hf_model_name).half()
        model = FSDP(
            model,
            cpu_offload=CPUOffload(offload_params=True),
            device_id=dist.get_rank()
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        for seq in tqdm(seqs):
            nlls = get_one_shot_log_likelihood(
                model,
                tokenizer,
                [seq],
                eps=1e-3
            )
            all_lls[col_name].append(nlls[0])
        
        # save all_lls to a pickle file if local rank is 0
        if dist.get_rank() == 0:
            torch.save(all_lls, f'wt_probs/{col_name}.pt')


if __name__ == "__main__":
    main()