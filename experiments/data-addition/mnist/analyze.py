import argparse

import torch
from kronfluence.analyzer import Analyzer, prepare_model
from kronfluence.arguments import FactorArguments
from kronfluence.utils.dataset import DataLoaderKwargs

from bio_if.tasks import ClassificationTask
from mnist import get_dataset, get_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--factor-strategy", type=str, default="ekfac")
    parser.add_argument("--query-batch-size", type=int, default=256)
    args = parser.parse_args()
    return args

def compute_influence(
        analyzer: Analyzer, 
        candidate_dataset: torch.utils.data.Dataset, 
        eval_dataset: torch.utils.data.Dataset, 
        args: argparse.Namespace
    ):
    # Configure parameters for DataLoader.
    dataloader_kwargs = DataLoaderKwargs(num_workers=4)
    analyzer.set_dataloader_kwargs(dataloader_kwargs)

    # Compute influence factors.
    factor_args = FactorArguments(strategy=args.factor_strategy)
    analyzer.fit_all_factors(
        factors_name=args.factor_strategy,
        dataset=candidate_dataset,
        per_device_batch_size=None,
        factor_args=factor_args,
        overwrite_output_dir=False,
    )
    # Compute pairwise scores.
    analyzer.compute_pairwise_scores(
        scores_name=args.factor_strategy,
        factors_name=args.factor_strategy,
        query_dataset=eval_dataset,
        train_dataset=candidate_dataset,
        per_device_query_batch_size=args.query_batch_size,
        overwrite_output_dir=False,
    )
    

def main():
    args = parse_args()
    # load model and prepare it for influence analysis
    task = ClassificationTask()
    model = get_model()
    model.load_state_dict(torch.load(f"models/mnist_pretrained_seed={args.seed}.pth"))
    model = prepare_model(model, task)

    if torch.cuda.is_available():
        model = model.cuda()

    # load datasets
    candidate_dataset = get_dataset("candidates", seed=args.seed)
    eval_dataset = get_dataset("validation", seed=args.seed)

    # create analyzer
    analyzer = Analyzer(
        analysis_name=f"mnist_seed={args.seed}",
        model=model,
        task=task,
    )

    # compute influence
    compute_influence(analyzer, candidate_dataset, eval_dataset, args)

    

if __name__ == "__main__":
    main()