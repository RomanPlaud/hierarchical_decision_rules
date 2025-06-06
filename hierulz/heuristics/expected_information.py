from .base_heuristic import Heuristic
import numpy as np

class ExpectedInformation(Heuristic):
    def __init__(self, hierarchy, _lambda=0.):
        """
        Initialize the ExpectedInformation heuristic with a hierarchy object.
        """
        super().__init__(hierarchy)
        self._lambda = _lambda  

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors using expected information.
        """
        predicted_nodes = np.argmax(proba_nodes*(self.hierarchy.information_content + self._lambda), axis=1)
        full_predictions = np.zeros_like(proba_nodes)
        for i, node in enumerate(predicted_nodes):
            full_predictions[i, self.hierarchy.get_ancestors(node)] = 1
        return full_predictions

        
        