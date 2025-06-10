from torchvision import transforms

from hierulz.datasets.dataset import HierarchicalDataset


def get_default_transform(dataset: str):
    """
    Return default transforms for a given dataset.
    """
    if (dataset.lower() == 'tieredimagenet') or (dataset.lower() == 'tieredimagenet_tiny'):
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif dataset.lower() == 'inat19':
        mean, std = [0.454, 0.474, 0.367], [0.237, 0.230, 0.249]
    else:
        raise ValueError(f"Unsupported dataset '{dataset}'")

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_dataset(dataset: str, split: str = 'test', transform=None, blurr_level=None):
    """
    Load a hierarchical dataset for inference.

    Args:
        dataset (str): Name of the dataset to use ('tieredimagenet' or 'inat19').
        split (str): Which split to load ('train', 'val', 'test').
        transform (Callable, optional): Transform pipeline. If None, default is used.
        blurr_level (float, optional): Gaussian blur sigma. If None, no blur applied.

    Returns:
        HierarchicalDataset: Initialized dataset.
    """
    root = f'data/datasets/{dataset}/{split}'

    if transform is None:
        transform = get_default_transform(dataset)

    return HierarchicalDataset(root=root, transform=transform, blurr_level=blurr_level)