import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", type=bool, default=False, help="Specify whether training or not")

    args = parser.parse_args()

    return args
