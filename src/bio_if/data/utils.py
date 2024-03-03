# import torch data utilities
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader


class FastaDataset(Dataset):
    def __init__(
        self,
        fasta_file: str,
        split: str = "train",
        tokenizer_fn: Callable = None,
    ):
        """
        A dataset for fasta files of format
        >SequenceN TARGET=value SET=[train,test] VALIDATION=bool
        sequence
        ...

        Args:
            fasta_file: str, path to fasta file
            split: str, one of "train", "val", "test"
            tokenizer_fn: Callable, function to tokenize sequences
        """
        self.fasta_file = fasta_file
        assert split in ["train", "val", "test"]
        self.split = split
        self.tokenizer_fn = tokenizer_fn
        self.seqs = []
        self.labels = []
        self._parse_fasta()

    def _parse_fasta(self):
        with open(self.fasta_file, "r") as f:
            # fasta of format
            # >SequenceN TARGET=value SET=[train,test] VALIDATION=bool
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                # ensure in split
                if self._header_to_split(lines[i]) == self.split:
                    # add data to seqs and labels
                    seq = lines[i + 1].strip()
                    label = self._header_to_label(lines[i])
                    self.seqs.append(seq)
                    self.labels.append(label)

        self.labels = torch.tensor(self.labels)

    def _header_to_split(self, header) -> str:
        # header is of format
        # >SequenceN TARGET=value SET=[train,test] VALIDATION=bool
        # return whether it is train, test, or val
        set_value = header.split()[2].split("=")[1]
        validation_value = header.split()[3].split("=")[1]
        if set_value == "test":
            return "test"
        if validation_value == "True":
            return "val"
        return "train"

    def _header_to_label(self, header) -> float:
        # header is of format
        # >SequenceN TARGET=value SET=[train,test] VALIDATION=bool
        # we want to return the value of TARGET
        return float(header.split()[1].split("=")[1])

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]

    def collate_fn(self, batch):
        # batch is a list of tuples, we want to return a tuple of tensors
        seqs, labels = zip(*batch)
        if self.tokenizer_fn is not None:
            seqs = self.tokenizer_fn(seqs)
        return seqs, torch.stack(labels)

    def get_dataloader(
        self, batch_size: int, shuffle: bool = False, drop_last: bool = False
    ):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate_fn,
        )
