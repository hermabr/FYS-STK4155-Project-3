import os
import torch
import argparse
import numpy as np

from analysis import logistic, pytorch, ffnn, bias_variance, test_sizes

#  from cnn import keras

SEED = 42

if __name__ == "__main__":
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    parser = argparse.ArgumentParser(
        description="To run the bias-variance tradeoff experiment"
    )
    parser.add_argument(
        "-t",
        "--test_size",
        action="store_true",
        help="To run test size analysis for the different models",
    )
    parser.add_argument(
        "-l",
        "--logistic",
        action="store_true",
        help="To run the logistic regression analysis",
    )
    parser.add_argument(
        "-c",
        "--cnn",
        action="store_true",
        help="To run the pytorch and tensorflow cnn analyses",
    )
    #  parser.add_argument(
    #      "-t",
    #      "--tensorflow",
    #      action="store_true",
    #      help="To run the tensorflow cnn analysis",
    #  )
    parser.add_argument(
        "-f",
        "--ffnn",
        action="store_true",
        help="To run the ffnn analysis",
    )
    parser.add_argument(
        "-bv",
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

    #  for seed in range(1, 100):
    #      np.random.seed(seed)
    #      torch.manual_seed(seed)
    #      print(f"Seed: {seed}")

    args = parser.parse_args()
    if not any(vars(args).values()):
        parser.print_help()
        exit(0)

    if args.test_size or args.all:
        test_sizes.main()
    if args.logistic or args.all:
        if args.all:
            print("Analysis for logistic regression")
        logistic.main()
    if args.cnn or args.all:
        if args.all:
            print("Analysis for pytorch cnn")
        pytorch.main()
    #  if args.ffnn or args.all:
    #      if args.all:
    #          print("Analysis for ffnn")
    #      ffnn.main()
    if args.biasvariance or args.all:
        if args.all:
            print("Analysis for bias-variance tradeoff")
        bias_variance.main()
