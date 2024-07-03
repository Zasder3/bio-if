import torch
from torch import nn
from torch.nn import functional as F
from kronfluence import Task


BATCH_TYPE = dict[str, torch.Tensor]


class MLMTask(Task):
    def __init__(self, special_token_ids: list[int]):
        super().__init__()
        self.special_token_ids = special_token_ids

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

        # Mask out special tokens from loss computation.
        for special_token_id in self.special_token_ids:
            batch["attention_mask"] = batch["attention_mask"] * (batch["input_ids"] != special_token_id).float()

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

        # Mask out special tokens from loss computation.
        for special_token_id in self.special_token_ids:
            batch["attention_mask"] = batch["attention_mask"] * (batch["input_ids"] != special_token_id).float()

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
