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


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation with hierarchical models")
    parser.add_argument('--model_name', required=True, help="Path to JSON config file with model configuration")
    parser.add_argument('--metric_name', required=True, help="Metric to evaluate the model on, e.g., 'accuracy', 'fß_score'")
    parser.add_argument('--dataset', required=True, help="Name of the dataset to use, e.g., 'tieredimagenet', 'inat19'")
    parser.add_argument('--decoding_method', required=True, help="Decoding method to use, e.g., 'opt', 'argmax', 'thresholding'")
    parser.add_argument('--path_probas', default=None, help="Path to the directory containing predictions")
    parser.add_argument('--path_save', required=True, help="Path to save the evaluation results")

def main():
    args = parse_args()

    # Load model configuration from JSON
    model_config = get_model_config(args.model_name)
    metric_config = get_metric_config(args.metric_name, args.dataset)
    dataset_config = get_dataset_config(args.dataset)

    hierarchy = load_hierarchy(dataset_config['name'])

    metric = load_metric(metric_config['metric_name'], **metric_config.get('kwargs', {}))
    if args.decoding_method is "opt":
        decoding = metric
    else:
        decoding = load_heuristic(args.decoding_method, dataset_config['name'])

    # load probas and labels according to args.config['path_predictions'] and args.config['blurr_level']

    if args.path_probas:
        # load probas and labels from the specified path
        probas_path = args.path_probas 
        labels_path = args.path_probas.replace('probas', 'labels')
    else:
        model_name = model_config.get('model_name', 'unnamed_model')
        suffix = f"_blurr_{model_config.get('blurr_level')}" if model_config.get('blurr_level') is not None else ''
        load_dir = Path(model_config.get('path_predictions', 'predictions')) / model_config['dataset'] / model_name
        
        probas_path = load_dir / f'probas_{args.split}{suffix}.pkl'
        labels_path = load_dir / f'labels_{args.split}{suffix}.pkl'

    # Load probabilities and labels
    probas = pkl.load(open(probas_path, 'rb'))
    labels = pkl.load(open(labels_path, 'rb'))
    labels = hierarchy.leaf_events[labels]  # Convert labels to 1-hot encoding

    # get nodes probas
    probas_nodes = hierarchy.get_probas(probas)
    # Decode predictions
    y_pred = decoding.decode(probas_nodes)
    # Compute the average metric
    score = metric.compute_metric(labels, y_pred)
    print(f"Evaluation score: {score}")
    # Save the results
    # open the save directory if is exists, otherwise create it
    save_dir = Path(args.path_save)
    save_dir.mkdir(parents=True, exist_ok=True)
    # open the json save file and write on it
    with open(save_dir, 'w') as f:
        json.dump({'model_name': model_config.get('model_name', 'unnamed_model'),
                   'pretrained': model_config.get('pretrained', False),
                   'blurr_level': model_config.get('blurr_level', 0),
                   'metric': metric_config['metric_name'],
                   'decoding_method': args.decoding_method,
                   'score': score}, f, indent=4)


if __name__ == '__main__':
    main()
    





