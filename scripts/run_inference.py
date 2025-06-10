import argparse
import json
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm
import pickle as pkl

import sys
sys.path.append('.')  # or the absolute path to the project root


from hierulz.datasets import get_dataset
from hierulz.models import load_model




def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with hierarchical models")
    parser.add_argument('--dataset', required=True, help='Name of the dataset to use, either tieredimagenet or inat19')
    parser.add_argument('--config_model', required=True, help="Path to JSON config file with model configuration")
    parser.add_argument('--split', default='test', help='Dataset split to use for inference (e.g., "test", "val")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--blurr_level', type=float, default=None)
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use for inference, if available')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model configuration from JSON
    with open(args.config_model, 'r') as f:
        config_model = json.load(f)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    transform = config_model.get('transform', None)
    dataset = get_dataset(dataset=args.dataset, split=args.split, transform=transform, blurr_level=args.blurr_level)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(config_model=config_model)
    model.to(device)
    model.eval()

    probas, labels = [], []
    with torch.no_grad():
        for images, target in tqdm(dataloader, desc="Running inference"):
            images = images.to(device)
            output = model(images)
            probas.append(output.cpu().numpy())
            labels.append(target.numpy())

    probas = np.concatenate(probas, axis=0)
    labels = np.concatenate(labels, axis=0)

    # --- Improved saving logic ---
    model_name = config_model.get('model_name', 'unnamed_model')
    suffix = f"_blurr_{args.blurr_level}" if args.blurr_level is not None else ''
    save_dir = Path('results') / args.dataset / model_name
    save_dir.mkdir(parents=True, exist_ok=True)

    probas_path = save_dir / f'probas_{args.split}{suffix}.pkl'
    labels_path = save_dir / f'labels_{args.split}{suffix}.pkl'

    pkl.dump(probas, open(probas_path, 'wb'))
    pkl.dump(labels, open(labels_path, 'wb'))

    print(f"Saved probabilities to: {probas_path}")
    print(f"Saved labels to:       {labels_path}")


if __name__ == '__main__':
    main()
