import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

ESM_MAX_SEQ_LEN = 1022


class MLMDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, seqs: list[str]):
        self.tokenizer = tokenizer
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        tokens = self.tokenizer(self.seqs[idx], return_tensors="pt")
        for key in tokens:
            tokens[key] = tokens[key].squeeze(0)
        # make a deep copy of input ids
        tokens["labels"] = tokens["input_ids"].clone()

        # select 15% of tokens to mask
        train_indices = torch.randperm(tokens["input_ids"].numel())[
            : int(0.15 * tokens["input_ids"].numel())
        ]
        unmasked_indices = torch.ones_like(tokens["input_ids"]).bool()
        unmasked_indices[train_indices] = False
        # turn unmasked tokens to -100
        tokens["labels"][unmasked_indices] = -100

        # choose which indices to mask, substitute, or leave unchanged
        rand_uniform = torch.rand(train_indices.numel())
        mask_indices = train_indices[rand_uniform <= 0.8]  # 80% chance to mask
        sub_indices = train_indices[
            (rand_uniform > 0.8) & (rand_uniform <= 0.9)
        ]  # 10% chance to substitute
        # 10% chance to leave unchanged

        # mask tokens
        tokens["input_ids"][mask_indices] = self.tokenizer.mask_token_id
        # substitute tokens
        tokens["input_ids"][sub_indices] = torch.randint(
            4, 24, (sub_indices.numel(),)
        )  # assuming ESM2 20 amino acids

        return tokens


class MinimalSequenceDataset(Dataset):
    def __init__(self, tokenizer: AutoTokenizer, seqs: list[str]):
        self.tokenizer = tokenizer
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        if len(seq) > ESM_MAX_SEQ_LEN:
            # randomly select a subsequence
            start = torch.randint(0, len(seq) - ESM_MAX_SEQ_LEN + 1, (1,)).item()
            seq = seq[start : start + ESM_MAX_SEQ_LEN]
        return self.tokenizer(seq)


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM

    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    seqs = [
        "MLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQ",
        "MLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQMLMVQA",
    ]
    dataset = MinimalSequenceDataset(tokenizer, seqs)

    from transformers import DataCollatorForLanguageModeling
    from datasets import Dataset

    # dataset = Dataset.from_dict({"seqs": seqs})
    # # tokenize the dataset
    # dataset = dataset.map(
    #     lambda x: tokenizer(x["seqs"], return_tensors="pt"),
    #     batched=False,
    # )
    print(dataset)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer, mlm=True, return_tensors="pt"
        ),
    )
    batch = next(iter(loader))
    print(batch)
    outputs = model(**batch)
    print(outputs)
