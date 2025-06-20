import sys
import os
sys.path.append('.')

import argparse
import json
import pickle as pkl
from pathlib import Path

from hierulz.hierarchy import load_hierarchy
from hierulz.metrics import load_metric, get_metric_config
from hierulz.heuristics import load_heuristic
from hierulz.models import get_model_config
from hierulz.datasets import get_dataset_config
from hierulz.utils import load_probas_and_labels, get_output_paths, save_evaluation_results


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with hierarchical models")
    parser.add_argument('--model_name', required=True, help="Path to JSON config file with model configuration")
    parser.add_argument('--metric_name', required=True, help="Metric to evaluate the model on, e.g., 'accuracy', 'fß_score'")
    parser.add_argument('--dataset', required=True, help="Name of the dataset to use, e.g., 'tieredimagenet', 'inat19'")
    parser.add_argument('--split', default='test', help='Dataset split to use for evaluation (e.g., "test", "val")')
    parser.add_argument('--blurr_level', type=float, default=None, help='Blurr level for the model, if applicable')
    parser.add_argument('--decoding_method', required=True, help="Decoding method to use, e.g., 'opt', 'argmax', 'thresholding'")
    parser.add_argument('--path_save', required=True, help="Path to save the evaluation results")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load model configuration from JSON
    model_config = get_model_config(args.model_name)
    metric_config = get_metric_config(args.metric_name, args.dataset)

    hierarchy = load_hierarchy(args.dataset)

    metric = load_metric(metric_config['metric_name'], **metric_config.get('kwargs', {}))
    if args.decoding_method == "Optimal":
        decoding = metric
    else:
        decoding = load_heuristic(args.decoding_method, args.dataset)

    probas_path, labels_path = get_output_paths(args.dataset, args.model_name, args.blurr_level, args.split)

    probas, labels = load_probas_and_labels(probas_path, labels_path)
    labels = hierarchy.leaf_events[labels]

    # get nodes probas
    probas_nodes = hierarchy.get_probas(probas)
    # Decode predictions
    y_pred = decoding.decode(probas_nodes)
    # Compute the average metric
    score = metric.compute_metric(labels, y_pred)
    # Save the results
    save_evaluation_results(args.path_save, args, model_config, score)


if __name__ == '__main__':
    main()
    





