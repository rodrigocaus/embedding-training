import argparse
import os
import sys

sys.path.append(os.path.abspath(".."))

import cli


def train(args: argparse.Namespace):
    return cli.train.main(args.config_file)


def evaluate(args: argparse.Namespace):
    pass


if __name__ == "__main__":

    # create the top-level parser
    parser = argparse.ArgumentParser(
        prog="embedding",
        description="Train, evaluate and analyze sentence transformers for paraphrase mining"
    )
    subparsers = parser.add_subparsers(required=True)

    # create the parser for the "train" command
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument(
        '-f', '--config-file',
        dest="config_file",
        type=str,
        required=True
    )
    parser_train.set_defaults(func=train)

    # create the parser for the "evaluate" command
    parser_eval = subparsers.add_parser('evaluate')
    parser_eval.add_argument('model_name', type=str, help="model name or path")
    parser_eval.add_argument(
        '-d', '--dataset', dest='dataset', type=str, help="path to dataset")
    parser_eval.set_defaults(func=evaluate)

    args = parser.parse_args()
    if not hasattr(args, 'func'):
        parser.print_help()
        exit(1)

    args.func(args)
