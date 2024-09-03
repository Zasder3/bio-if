from matplotlib import pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import PreTrainedTokenizer, EsmForMaskedLM

# 1024 is the maximum sequence length for ESM-2, minus 2 for the BOS and EOS tokens
MODEL_WINDOW = 1022


def get_optimal_window(
    mutation_position_relative: int,
    seq_len_wo_special: int,
    seq: torch.tensor,
    tokenizer: PreTrainedTokenizer,
) -> tuple[torch.tensor, int]:
    # adapted from https://github.com/OATML-Markslab/ProteinGym/blob/main/proteingym/utils/scoring_utils.py
    initial_residue = seq[0, mutation_position_relative]
    seq = seq.clone()
    mutation_position_relative -= 1  # 1-indexed to 0-indexed
    half_model_window = MODEL_WINDOW // 2
    if seq_len_wo_special <= MODEL_WINDOW:
        mutation_position_relative += 1
        assert initial_residue == seq[0, mutation_position_relative]
        seq[0, mutation_position_relative] = tokenizer.mask_token_id
        return seq, mutation_position_relative
    elif mutation_position_relative < half_model_window:
        mutation_position_relative += 1
        seq = seq[:, : MODEL_WINDOW + 1]
        seq = torch.cat(
            [
                seq,
                torch.tensor([[tokenizer.eos_token_id]], device=seq.device),
            ],
            dim=1,
        )
        assert initial_residue == seq[0, mutation_position_relative]
        seq[0, mutation_position_relative] = tokenizer.mask_token_id
        return seq, mutation_position_relative
    elif mutation_position_relative >= seq_len_wo_special - half_model_window:
        mutation_position_relative += 1
        mutation_position_relative -= seq_len_wo_special - MODEL_WINDOW
        seq = seq[:, -MODEL_WINDOW - 1 :]
        seq = torch.cat(
            [
                torch.tensor([[tokenizer.cls_token_id]], device=seq.device),
                seq,
            ],
            dim=1,
        )
        assert initial_residue == seq[0, mutation_position_relative]
        seq[0, mutation_position_relative] = tokenizer.mask_token_id
        return seq, mutation_position_relative
    else:
        seq = seq[
            :,
            max(0, mutation_position_relative - half_model_window) + 1 : min(
                seq_len_wo_special, mutation_position_relative + half_model_window
            )
            + 1,  # +1 for BOS token
        ]
        mutation_position_relative -= max(
            0, mutation_position_relative - half_model_window
        )
        mutation_position_relative += 1
        seq = torch.cat(
            [
                torch.tensor([[tokenizer.cls_token_id]], device=seq.device),
                seq,
                torch.tensor([[tokenizer.eos_token_id]], device=seq.device),
            ],
            dim=1,
        )
        assert initial_residue == seq[0, mutation_position_relative], (
            initial_residue,
            seq[0, mutation_position_relative - 2 : mutation_position_relative + 3],
        )
        seq[0, mutation_position_relative] = tokenizer.mask_token_id
        return seq, mutation_position_relative


def load_study_data(
    study_data_dir: str,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> tuple[list[list[tuple[int, int]]], list[float], torch.tensor]:
    """
    Load study data from study data directory.

    Args:
        study_data_dir: Directory containing study data.
    Returns:
        List of mutations, list of fitness values, and the wildtype sequence.
    """
    # load study data
    study_df = pd.read_csv(study_data_dir)

    # create list of mutations and fitness values
    mutations = study_df["mutant"].tolist()
    fitness_values = study_df["DMS_score"].tolist()

    # mutants are in the form of "{old_aa}{position}{new_aa}:{old_aa}{position}{new_aa}..."
    # preprocess them into just "[({position}, {new_aa_tokenized}), ...]"
    mutations = [m.split(":") for m in mutations]  # ["{old_aa}{position}{new_aa}", ...]
    mutations = [
        [(int(m[1:-1]), tokenizer.convert_tokens_to_ids(m[-1])) for m in mut]
        for mut in mutations
    ]  # [({position}, {new_aa_tokenized}), ...]

    # recover wildtype sequence using the first mutant
    wt = tokenizer(study_df["mutated_sequence"].iloc[0], return_tensors="pt")[
        "input_ids"
    ].to(device)
    init_res = [
        tokenizer.convert_tokens_to_ids(mut[0])
        for mut in study_df["mutant"].iloc[0].split(":")
    ]
    for init_res, (pos, new_aa) in zip(init_res, mutations[0]):
        # because pos is 1-indexed we don't need to subtract 1 from it as we now have the BOS token
        assert (
            wt[0, pos] == new_aa
        ), f"Expected {init_res} at position {pos}, got {wt[0, pos]}"
        wt[0, pos] = init_res

    return mutations, fitness_values, wt


def evaluate_zero_shot_fitness_prediction(
    model: EsmForMaskedLM, tokenizer: PreTrainedTokenizer, fitness_dataset_path: str
) -> float:
    # load study data
    mutations, fitness_values, wt = load_study_data(
        fitness_dataset_path, tokenizer, model.device
    )

    mutation_idx_set = set()
    for mut in mutations:
        for pos, _ in mut:
            mutation_idx_set.add(pos)

    # evaluate masked marginals at each position
    marginal_differences = {}

    for mutation_idx in tqdm(mutation_idx_set, desc="Processing mutations"):
        # get the optimal window around the mutation
        optimal_window_seq, new_mutation_idx = get_optimal_window(
            mutation_idx, wt.size(1) - 2, wt, tokenizer
        )

        # get the masked marginals
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output = model(input_ids=optimal_window_seq)
            # because we're calculating a log odds ratio we don't need to use logsumexp to normalize
            marginal_differences[mutation_idx] = (
                output["logits"][0, new_mutation_idx]
                - output["logits"][0, new_mutation_idx][wt[0, mutation_idx]]
            )
    # translate each mutation to a fitness prediction by summing the marginal differences of the new amino acids
    fitness_predictions = [
        sum(marginal_differences[pos][new_aa].item() for pos, new_aa in mut)
        for mut in mutations
    ]

    # scatter plot of fitness values vs fitness predictions
    plt.scatter(fitness_values, fitness_predictions)
    plt.savefig("fitness_values_vs_predictions.png")
    # calculate spearman correlation
    spearman_correlation, _ = spearmanr(fitness_values, fitness_predictions)

    return spearman_correlation


if __name__ == "__main__":
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    # load model
    model = (
        AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
        .to("cuda:0")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

    # evaluate fitness prediction
    # fitness_dataset_dir = "experiments/dms/studies/A4GRB6_PSEAI_Chen_2020.csv"
    fitness_dataset_dir = "experiments/dms/studies/A0A1I9GEU1_NEIME_Kennouche_2019.csv"
    print(evaluate_zero_shot_fitness_prediction(model, tokenizer, fitness_dataset_dir))
