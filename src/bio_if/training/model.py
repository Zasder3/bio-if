from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    EsmForMaskedLM,
)


def load_esm_model(model_name: str) -> tuple[EsmForMaskedLM, PreTrainedTokenizer]:
    model: EsmForMaskedLM = AutoModelForMaskedLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    del model.esm.contact_head
    return model, tokenizer
