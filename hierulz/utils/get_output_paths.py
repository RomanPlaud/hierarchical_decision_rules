from pathlib import Path

def get_output_paths(dataset, model_name=None, blurr_level=None, split='val'):
    """
    Generate file paths for saving probas and labels based on dataset and model information.

    Args:
        dataset (str): Name of the dataset.
        model_name (str): Model name (default: 'unnamed_model' if None).
        blurr_level (int or None): Optional blurring level.
        split (str): Data split name ('train', 'val', 'test').

    Returns:
        tuple: (probas_path, labels_path) as Path objects.
    """
    model_name = model_name or 'unnamed_model'
    suffix = f"_blurr_{blurr_level}" if blurr_level is not None else ''
    save_dir = Path('results') / dataset / model_name
    probas_path = save_dir / f'probas_{split}{suffix}.pkl'
    labels_path = save_dir / f'labels_{split}{suffix}.pkl'

    return probas_path, labels_path
