import pickle as pkl
from .hierarchy import Hierarchy

def load_hierarchy(path):
    """
    Load a hierarchy from a pickle file.

    Args:
        path (str): Path to the pickle file containing the hierarchy.

    Returns:
        dict: The loaded hierarchy.
    """
    with open(path, 'rb') as f:
        hierarchy = pkl.load(f)
    
    return Hierarchy(hierarchy)