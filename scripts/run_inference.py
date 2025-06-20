import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl

import sys


from hierulz.datasets import get_dataset, get_default_transform
from hierulz.models import load_model, get_model_config
from hierulz.utils import get_output_paths, save_probas_labels




def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with hierarchical models")
    parser.add_argument('--dataset', required=True, help='Name of the dataset to use, either tieredimagenet or inat19')
    parser.add_argument('--model_name', required=True, help="Name of the model to load, e.g., 'alexnet', 'resnet18'")
    parser.add_argument('--split', default='test', help='Dataset split to use for inference (e.g., "test", "val")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--blurr_level', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use for inference, if available')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model configuration from JSON

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    transform = get_default_transform(args.dataset)
    dataset = get_dataset(dataset=args.dataset, split=args.split, transform=transform, blurr_level=args.blurr_level)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    config_model = get_model_config(args.model_name)
    model = load_model(config_model=config_model)
    model.to(device)
    model.eval()

    probas, labels = [], []
    with torch.no_grad():
        for images, target in tqdm(dataloader, 
        desc=print(f"Running inference on {args.dataset} [{args.split}] with model from {args.model_name} for blurr level {args.blurr_level}")):
            images = images.to(device)
            output = model(images)
            probas.append(output.cpu().numpy())
            labels.append(target.numpy())

    probas = np.concatenate(probas, axis=0)
    labels = np.concatenate(labels, axis=0)

    probas_path, labels_path = get_output_paths(
        dataset=args.dataset,
        model_name=args.model_name,
        blurr_level=args.blurr_level,
        split=args.split
    )
    print(f"Saving probabilities to {probas_path} and labels to {labels_path}")

    save_probas_labels(probas, labels, probas_path, labels_path)


if __name__ == '__main__':
    main()
