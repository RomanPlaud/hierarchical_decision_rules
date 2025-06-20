import pickle as pkl

def save_probas_labels(probas, labels, probas_path: str, labels_path: str):
    pkl.dump(probas, open(probas_path, 'wb'))
    pkl.dump(labels, open(labels_path, 'wb'))