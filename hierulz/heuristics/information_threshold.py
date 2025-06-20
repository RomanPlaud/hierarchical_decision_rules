from .base_heuristic import Heuristic
import numpy as np

class InformationThreshold(Heuristic):
    def __init__(self, hierarchy, threshold=0.0):
        """
        Initialize the InformationThreshold heuristic with a hierarchy object and a threshold.
        """
        super().__init__(hierarchy)
        self.threshold = threshold

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Given node-wise probabilities, select the most probable node whose information content exceeds the threshold.
        Return binary vectors with 1s for the selected node and all its ancestors.
        """
        candidates = self.hierarchy.information_content >= self.threshold
        predicted_nodes = np.argmax(candidates*proba_nodes, axis=1)
        full_predictions = np.zeros_like(proba_nodes)
        for i, node in enumerate(predicted_nodes):
            full_predictions[i, self.hierarchy.get_ancestors(node)] = 1
        return full_predictions