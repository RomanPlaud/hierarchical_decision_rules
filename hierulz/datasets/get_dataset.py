# get_dataset(dataset=args.dataset, split=args.split, blurr_level=args.blurr_level)
from hierulz.datasets.dataset import HierarchicalDataset



def get_dataset(dataset, split='test', blurr_level=None, kernel_size=61):
    """
    Get a dataset for inference.

    Args:
        dataset (str): Name of the dataset to use, either 'tieredimagenet' or 'inat19'.
        split (str): Dataset split to use for inference (e.g., 'test', 'val').
        blurr_level (float, optional): Sigma value for Gaussian blur. If None, no blur is applied.

    Returns:
        HierarchicalDataset: The dataset object.
    """
    root = f'data/datasets/{dataset}/{split}'
    return HierarchicalDataset(root=root, blurr_level=blurr_level)