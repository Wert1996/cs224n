#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run inference for one sentence with a trained parser checkpoint.
"""
import argparse

import torch

from parser_model import ParserModel
from utils.parser_utils import load_and_preprocess_data


def build_inference_example(parser, tokens):
    # Placeholder fields are required by parser.vectorize.
    example = {
        "word": tokens,
        "pos": ["NN"] * len(tokens),
        "head": [0] * len(tokens),
        "label": [parser.root_label] * len(tokens),
    }
    return parser.vectorize([example])


def main():
    argp = argparse.ArgumentParser(description="Run parser inference on one sentence.")
    argp.add_argument(
        "--weights",
        required=True,
        help="Path to model.weights produced during training.",
    )
    argp.add_argument(
        "--sentence",
        required=True,
        help='Raw input sentence, e.g. "I love NLP".',
    )
    argp.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Use reduced preprocessing to match debug-trained checkpoints.",
    )
    args = argp.parse_args()

    parser, embeddings, _, _, _ = load_and_preprocess_data(args.debug)
    model = ParserModel(embeddings)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    parser.model = model

    tokens = args.sentence.strip().split()
    if not tokens:
        raise ValueError("Sentence is empty after tokenization.")

    dataset = build_inference_example(parser, tokens)
    uas, dependencies = parser.parse(dataset, eval_batch_size=1)

    print("Tokens:")
    print(tokens)
    print("\nDependencies (head -> dependent):")
    for head_idx, dep_idx in dependencies[0]:
        head = "ROOT" if head_idx == 0 else tokens[head_idx - 1]
        dep = tokens[dep_idx - 1]
        print(f"{head} -> {dep}")
    print(f"\nUAS (not meaningful for unlabeled custom input): {uas:.4f}")


if __name__ == "__main__":
    main()
