import argparse
import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from tqdm import tqdm

from bio_if.data.utils import FastaDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Embed fasta files")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to input fasta file",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=os.path.expanduser("~/.cache/bio_if"),
        help="Path to output file",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to ESM model",
    )
    parser.add_argument(
        "-n",
        "--dataset_name",
        type=str,
        required=True,
        help="Name of dataset",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )

    return parser.parse_args()


def embed_fasta(args):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForMaskedLM.from_pretrained(args.model).to(args.device)
    # load fasta file
    tokenizer_fn = lambda x: tokenizer(x, return_tensors="pt")["input_ids"]
    dataset = FastaDataset(args.input, split=None, tokenizer_fn=tokenizer_fn)
    loader = dataset.get_dataloader(batch_size=args.batch_size, shuffle=False)
    # embed sequences
    seq_to_embed = {}
    for i, (seq, _) in tqdm(enumerate(loader), total=len(loader)):
        seq = seq.to(args.device)
        with torch.no_grad():
            outputs = model(seq, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][:, 0, :]
        for j in range(len(seq)):
            seq_to_embed[dataset.seqs[j + i * args.batch_size]] = embeddings[
                j
            ].cpu()
    # save to output_dir
    out_file = os.path.join(
        args.output_dir, args.model, f"{args.dataset_name}.pt"
    )
    # ensure directory exists
    os.makedirs(os.path.join(args.output_dir, args.model), exist_ok=True)
    torch.save(seq_to_embed, out_file)


def main():
    args = parse_args()
    embed_fasta(args)


if __name__ == "__main__":
    main()
