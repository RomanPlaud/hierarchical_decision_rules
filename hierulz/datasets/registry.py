
# Map dataset names to their config JSON files (full or relative paths)
DATASET_REGISTRY = {
    "tieredimagenet": "configs/datasets/config_tieredimagenet.json",
    "tieredimagenet_tiny": "configs/datasets/config_tieredimagenet_tiny.json",
    "inat19": "configs/datasets/config_inat19.json",
}

def get_config_path(dataset_name):
    """
    Given a dataset name, returns the path to its config JSON file.
    Raises KeyError if dataset not found.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise KeyError(f"Dataset '{dataset_name}' is not registered.")
    return DATASET_REGISTRY[dataset_name]
