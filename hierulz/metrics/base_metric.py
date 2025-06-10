import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, hierarchy):
        """
        Common initializer: Initialize the metric with a hierarchy object.
        """    
        self.hierarchy = hierarchy
        self.leaf_events = hierarchy.leaf_events

    @abstractmethod
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute a “distance” or “loss”/“score” between a single true‐event vector
        (length‐n_nodes binary) and a single predicted event vector (length‐n_nodes).
        Must return a float.
        """
        pass

    @abstractmethod
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Given node‐wise predictions of shape `(n_samples, n_nodes)`, produce
        a boolean/binary (0/1) vector for each sample of shape `(n_samples, n_nodes)`.
        """
        pass
    
    def evaluate(self, p_leaves_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Generic evaluation:
          • p_leaves_true: shape (n_samples, n_leaves), a probability distribution over leaf events.
                            Each row should sum to 1.
          • y_pred: either
              - a binary matrix of shape (n_samples, n_nodes) 
              - a continuous vector of shape (n_samples, n_nodes) that will be decoded to binary with `self.decode(...)`.
        
        Returns: a length‐n_samples vector, where each entry is
          Σ_{j=0..n_leaves−1} p_leaves_true[i, j] * metric(leaf_events[j], y_pred_i)
        """
        # If y_pred is not already {0,1}, threshold it.
        # We check “is it boolean/binary?” by seeing if all entries are 0/1
        if not np.array_equal(y_pred, y_pred.astype(bool)):
            y_pred_bin = self.decode(y_pred)
        else:
            y_pred_bin = y_pred.astype(int)

        n_samples = y_pred_bin.shape[0]
        n_events  = self.leaf_events.shape[0]
        
        # Build a matrix of shape (n_samples, n_events) of pairwise metric(...) calls
        values = np.zeros((n_samples, n_events), dtype=float)
        for i in range(n_samples):
            for j in range(n_events):
                # compare true‐event vector j (self.leaf_events[j]) vs. predicted vector y_pred_bin[i]
                values[i, j] = self.metric(self.leaf_events[j], y_pred_bin[i])

        # Finally, do the weighted sum over the true‐event probabilities
        return (p_leaves_true * values).sum(axis=1)
    
    def compute_metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the average metric value for a batch of true and predicted event vectors.
        :param y_true: True event vectors of shape (n_samples, n_nodes).
        :param y_pred: Predicted event vectors of shape (n_samples, n_nodes).
        :return: Average metric value.
        """
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")
        
        n_samples = y_true.shape[0]
        total_metric = 0.0
        
        for i in range(n_samples):
            total_metric += self.metric(y_true[i], y_pred[i])
        
        return total_metric / n_samples
    
    
    def check_if_y_pred_is_a_single_node(self, y_pred):
        """
        Check if the prediction is for a single node.
        This is used to ensure that the node metric can be computed correctly.
        """

        depths_pred = self.hierarchy.depths * y_pred

        return (np.bincount(depths_pred)[1:].max() == 1)