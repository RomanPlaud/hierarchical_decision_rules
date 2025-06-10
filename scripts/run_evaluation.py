import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with hierarchical models")
    parser.add_argument('--config', required=True, help="Path to JSON config file with model configuration")
    parser.add_argument('--metric', help="Metric to evaluate the model on, e.g., 'accuracy', 'fß_score'")
    parser.add_argument('--decoding_method', help="Decoding method to use, e.g., 'opt', 'argmax', 'thresholding'")


def main():


