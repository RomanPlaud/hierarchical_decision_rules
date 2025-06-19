import pickle as pkl
from .hierarchy import Hierarchy
from hierulz.datasets import get_dataset_config

def load_hierarchy(name: str) -> Hierarchy:
    """
    Load a hierarchy from a pickle file.

    Args:
        path (str): Path to the pickle file containing the hierarchy.

    Returns:
        dict: The loaded hierarchy.
    """
    dataset_config = get_dataset_config(name)
    with open(dataset_config['hierarchy_idx'], 'rb') as f:
        hierarchy = pkl.load(f)
    
    return Hierarchy(hierarchy)