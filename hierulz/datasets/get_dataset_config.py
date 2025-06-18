import json
from .registry import get_config_path

def get_dataset_config(dataset_name):
    """
    Loads and returns the dataset config dictionary by dataset name.
    """
    config_path = get_config_path(dataset_name)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
