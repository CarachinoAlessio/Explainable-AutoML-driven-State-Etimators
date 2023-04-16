import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", type=bool, default=False, help="Specify whether training or not")
    parser.add_argument("--shap_values", type=bool, default=False, help="Specify whether re-compute and save shap values")
    parser.add_argument("--verbose", type=bool, default=False, help="Specify whether to print more logs than necessary")
    parser.add_argument("--s", type=int, default=1, help="Specify the number of times it will retrain")

    args = parser.parse_args()

    return args
