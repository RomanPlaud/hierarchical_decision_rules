from pathlib import Path
import pickle as pkl

def load_probas_and_labels(probas_path, labels_path):
    """Load probability and label files"""

    with open(probas_path, 'rb') as f:
        probas = pkl.load(f)
    with open(labels_path, 'rb') as f:
        labels = pkl.load(f)

    return probas, labels
