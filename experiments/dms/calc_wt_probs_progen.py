from transformers import AutoModelForCausalLM
from tokenizers import Tokenizer
from tqdm import tqdm
import pandas as pd
import torch


def cross_entropy(logits, target, reduction="mean"):
    return torch.nn.functional.cross_entropy(
        input=logits,
        target=target,
        weight=None,
        size_average=None,
        reduce=None,
        reduction=reduction,
    )


def log_likelihood(logits, target, reduction="mean"):
    return -cross_entropy(
        logits.view(-1, logits.size(-1)), target.view(-1), reduction=reduction
    )


def get_log_likelihood(model, seq):
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            # get log likelihood of the sequence
            logits = model(seq[:-1]).logits
            labels = seq[1:]

            # remove unused logits
            first_token, last_token = 5, 29
            logits = logits[:, first_token : last_token + 1]
            labels = labels - first_token

        return log_likelihood(logits, labels).item()


def main():
    df = pd.read_csv("DMS_substitutions.csv")

    seqs = df["target_seq"].tolist()
    max_positions = max([len(seq) for seq in seqs])
    all_lls = {}
    model_names = [
        ("hugohrban/progen2-small", "Progen2 S Log Likelihood"),
        ("hugohrban/progen2-medium", "Progen2 M Log Likelihood"),
        ("hugohrban/progen2-large", "Progen2 L Log Likelihood"),
        ("hugohrban/progen2-xlarge", "Progen2 XL Log Likelihood"),
    ]
    for hf_model_name, col_name in model_names:
        print(col_name)
        all_lls[col_name] = []

        model = (
            AutoModelForCausalLM.from_pretrained(hf_model_name, trust_remote_code=True)
            .to("cuda:0")
            .half()
        )
        # fix the attention mask to account for the longer context
        for layer in model.transformer.h:
            layer.attn.bias = torch.tril(
                torch.ones(
                    (max_positions, max_positions),
                    dtype=torch.bool,
                    device=model.device,
                )
            ).view(1, 1, max_positions, max_positions)
        tokenizer = Tokenizer.from_pretrained(hf_model_name)
        for seq in tqdm(seqs):
            seq = "1" + seq + "2"
            input_fwd = torch.tensor(tokenizer.encode(seq[:-1]).ids).to(model.device)
            input_bwd = torch.tensor(tokenizer.encode(seq[::-1][:-1]).ids).to(
                model.device
            )
            nll_fwd = get_log_likelihood(
                model,
                input_fwd,
            )
            nll_bwd = get_log_likelihood(
                model,
                input_bwd,
            )
            nll = (nll_fwd + nll_bwd) / 2
            all_lls[col_name].append(nll)

        torch.save(all_lls, f"wt_probs/{col_name}.pt")


if __name__ == "__main__":
    main()
