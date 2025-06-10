from .base_metric import Metric
import numpy as np

class Accuracy(Metric):

    def __init__(self, hierarchy):
        super().__init__(hierarchy)

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.all(y_true == y_pred))
    
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors.
        Here, optimal decoding is to take the argmax over the leaf nodes
        """
        p_leaves = p_nodes[:, self.hierarchy.leaves_idx]
        leaf_preds = np.argmax(p_leaves, axis=1)
        y_pred = self.leaf_events[leaf_preds]
        return y_pred