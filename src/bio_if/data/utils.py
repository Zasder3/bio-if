import os
from typing import Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader

GB1_IDX = [38, 39, 40, 53]


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
        assert split in ["train", "val", "test", None]
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
                if (
                    self._header_to_split(lines[i]) == self.split
                    or self.split is None
                ):
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


class CachedEmbedTokenizer:
    def __init__(self, cache: str):
        """
        Load fixed length cached embeddings as a way to tokenize sequences.
        Args:
            cache: str, path to cache
        """
        self.cache = torch.load(os.path.expanduser(cache))

    def __call__(self, seq=Union[str, list[str]]):
        """
        Tokenize a sequence or list of sequences.

        Args:
            seq: Union[str, list[str]], sequence or list of sequences
        Returns:
            torch.Tensor, tokenized sequence
        """
        if isinstance(seq, str):
            return self.cache[seq]
        return torch.stack([self.cache[s] for s in seq])


class FixedIndexTokenizer:
    def __init__(self, indices: list[int] = GB1_IDX):
        self.indices = indices
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    def __call__(self, seq: Union[str, list[str]]):
        """
        Tokenize a sequence or list of sequences.

        Args:
            seq: Union[str, list[str]], sequence or list of sequences
        Returns:
            torch.Tensor, tokenized sequence
        """
        if isinstance(seq, str):
            return self._tokenize(seq)
        return torch.stack([self._tokenize(s) for s in seq])

    def _tokenize(self, seq: str) -> torch.Tensor:
        """
        Tokenize a sequence.

        Args:
            seq: str, sequence
        Returns:
            torch.Tensor, tokenized sequence
        """
        # convert to one hot
        return torch.eye(20)[
            torch.tensor([self.amino_acids.index(seq[i]) for i in self.indices])
        ].reshape(-1)
