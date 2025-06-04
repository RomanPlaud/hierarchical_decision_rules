from base_metric import Metric
import numpy as np

class HammingLoss(Metric):
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the Hamming loss between the true and predicted event vectors.
        Hamming loss is the fraction of labels that are incorrectly predicted.
        """
        return float(np.mean(y_true != y_pred))
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors.
        Here, optimal decoding is to threshold the probabilities at 0.5.
        """
        return (p_nodes > 0.5).astype(int)