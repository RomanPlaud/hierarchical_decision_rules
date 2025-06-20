import pickle as pkl
from pathlib import Path

def save_probas_labels(probas, labels, probas_path: Path, labels_path: Path):
    # create directories if they do not exist
    probas_dir = probas_path.parent
    labels_dir = labels_path.parent
    if not probas_dir.exists() or not labels_dir.exists():
        print(f"Creating directories: {probas_dir} and {labels_dir}")
        probas_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
    # save probas and labels
    pkl.dump(probas, open(probas_path, 'wb'))
    pkl.dump(labels, open(labels_path, 'wb'))