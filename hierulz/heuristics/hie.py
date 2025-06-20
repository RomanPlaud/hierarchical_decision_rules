from .base_heuristic import Heuristic
import numpy as np

class HiE(Heuristic):
    def __init__(self, hierarchy):
        """
        Initialize the HiE heuristic with a hierarchy object and a confidence threshold.
        """
        super().__init__(hierarchy)

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors using the HiE heuristic.
        See https://openreview.net/forum?id=hiQG8qGxso for details.
        """
        proba_leaves = proba_nodes[:, self.hierarchy.leaves_idx]
        probas_parents = proba_nodes[:, [self.hierarchy.parent[l] for l in self.hierarchy.leaves_idx]]

        predicted_nodes = np.argmax(proba_leaves * probas_parents, axis=1)
        full_predictions = np.zeros_like(proba_nodes)
        
        for i, node_predicted in enumerate(predicted_nodes):
            full_predictions[i] = self.hierarchy.leaf_events[node_predicted]
            # Set the corresponding indices in full_predictions to 1

        return full_predictions