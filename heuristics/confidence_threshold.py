from .base_heuristic import Heuristic
import numpy as np

class ConfidenceThreshold(Heuristic):
    def __init__(self, hierarchy, threshold=0.5):
        """
        Initialize the ConfidenceThreshold heuristic with a hierarchy object and a confidence threshold.
        """
        super().__init__(hierarchy)
        self.threshold = threshold

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors based on a confidence threshold.
        """
        candidates = proba_nodes >= self.threshold
        predicted_nodes = np.argmax(candidates*self.hierarchy.information_content, axis=1)
        full_predictions = np.zeros_like(proba_nodes)
        for i, node in enumerate(predicted_nodes):
            full_predictions[i, self.hierarchy.get_ancestors(node)] = 1
        return full_predictions
        