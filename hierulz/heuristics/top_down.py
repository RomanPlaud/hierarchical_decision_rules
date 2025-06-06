from .base_heuristic import Heuristic
import numpy as np

class TopDown(Heuristic):
    def __init__(self, hierarchy):
        """
        Initialize the TopDown heuristic with a hierarchy object.
        """
        super().__init__(hierarchy)

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors using a top-down approach.
        Here, optimal decoding is to take the argmax recursively from root to a leaf node.
        """
        def decode_rec(node, cand, p):
            if node in self.hierarchy.leaves_idx:
                cand[node] = 1
                return cand
            else :
                cand[node] = 1
                children = self.hierarchy.hierarchy_dico_idx[node]
                maxi_node = children[np.argmax(p[children])]
                decode_rec(maxi_node, cand, p)

        full_predictions = np.zeros_like(proba_nodes)
        for p_i, cand_i in zip(proba_nodes, full_predictions):
            decode_rec(self.hierarchy.root_idx, cand_i, p_i)
        return full_predictions 