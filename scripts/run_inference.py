import argparse
import os
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from hierulz.datasets import get_dataset
from hierulz.hierarchy import load_hierarchy, Hierarchy
from hierulz.models import load_model, infer_on_batch
from hierulz.utils import save_pickle


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with hierarchical models")
    parser.add_argument('--dataset', required=True, help='Name of the dataset to use, either tieredimagenet or inat19')
    parser.add_argument('--config_model', required=True, help="json file with model configuration")

    parser.add_argument('--split', default='test', help='Dataset split to use for inference (e.g., "test", "val")')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--blurr_level', type=float, default=None)

    parser.add_argument('--gpu', type=int, default=0, help='GPU device index to use for inference, if available')

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dataset = get_dataset(dataset=args.dataset, split=args.split, blurr_level=args.blurr_level)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model = load_model(args.model, dataset_name=args.dataset, pretrained=args.pretrained, device=device)

    probas, labels = [], []
    model.eval()
    with torch.no_grad():
        for images, target in tqdm(dataloader):
            images = images.to(device)
            output = infer_on_batch(model, images, args.model, args.dataset)
            probas.append(output.cpu())
            labels.append(target)

    save_dir = os.path.join('results', args.dataset, args.model)
    os.makedirs(save_dir, exist_ok=True)
    suffix = f'_blurr_{args.blurr_level}' if args.blurr_level else ''

    save_pickle(probas, os.path.join(save_dir, f'probas_{args.split}{suffix}.pkl'))
    save_pickle(labels, os.path.join(save_dir, f'labels_{args.split}{suffix}.pkl'))


if __name__ == '__main__':
    main()
