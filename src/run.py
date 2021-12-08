import torch
import argparse
import numpy as np

from analysis import pytorch, bias_variance, compare_models

SEED = 42

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(
        description="To run the bias-variance tradeoff experiment"
    )
    parser.add_argument(
        "-m",
        "--models",
        action="store_true",
        help="To run test of different models to evaluate their performance",
    )
    parser.add_argument(
        "-c",
        "--cnn",
        action="store_true",
        help="To run the pytorch and tensorflow cnn analyses",
    )
    parser.add_argument(
        "-b",
        "--biasvariance",
        help="To test the bias-variance tradeoff for three different methods",
        action="store_true",
    )
    parser.add_argument(
        "-a",
        "--all",
        help="To run all the analyzes",
        action="store_true",
    )

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    if args.models or args.all:
        if args.all:
            print("Analysis for performance of different models")
        compare_models.main()
    if args.cnn or args.all:
        if args.all:
            print("Analysis for pytorch cnn")
        pytorch.main()
    if args.biasvariance or args.all:
        if args.all:
            print("Analysis for bias-variance tradeoff")
        bias_variance.main()
