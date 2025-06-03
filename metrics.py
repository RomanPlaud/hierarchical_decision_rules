import numpy as np
from abc import ABC, abstractmethod

class Metric(ABC):
    def __init__(self, hierarchy):
        """
        Common initializer: Initialize the metric with a hierarchy object.
        """    
        if not hasattr(hierarchy, 'events') or hierarchy.events is None:
            if hasattr(hierarchy, 'get_events'):
                hierarchy.get_events()
            else:
                raise AttributeError("Hierarchy object has no 'events' attribute or 'get_events' method.")
        self.leaf_events = hierarchy.events

    @abstractmethod
    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute a “distance” or “loss”/“score” between a single true‐event vector
        (length‐n_nodes binary) and a single predicted event vector (length‐n_nodes).
        Must return a float.
        """
        ...

    @abstractmethod
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Given node‐wise predictions of shape `(n_samples, n_nodes)`, produce
        a boolean/binary (0/1) vector for each sample of shape `(n_samples, n_nodes)`.
        """
        ...

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

class Accuracy(Metric):

    def __init__(self, hierarchy):
        super().__init__(hierarchy)
        if not hasattr(hierarchy, 'leaves') or hierarchy.leaves is None:
            if hasattr(hierarchy, 'get_leaves'):
                hierarchy.get_leaves()
            else:
                raise AttributeError("Hierarchy object has no leaves or get_leaves method.")
        
        self.leaves_idx = [hierarchy.leaf2i[l] for l in hierarchy.leaves]

    def metric(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.all(y_true == y_pred))
    
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors.
        Here, optimal decoding is to take the argmax over the leaf nodes
        """
        p_leaves = p_nodes[:, self.leaves_idx]
        leaf_preds = np.argmax(p_leaves, axis=1)
        y_pred = self.leaf_events[leaf_preds]
        return y_pred