from .base_metric import Metric
import numpy as np
import pickle


class Leaf2LeafMetric(Metric):
    def __init__(self, hierarchy, cost_matrix_path):
        super().__init__(hierarchy)
        self.cost_matrix = _load_cost_matrix(cost_matrix_path)
    
    def metric(self, y_true, y_pred):
        if not self.check_if_y_pred_is_a_single_leaf(y_pred):
            raise ValueError("internal node or multiple nodes is not supported in Leaf2LeafMetric")

        leaf_true = np.argmax(self.hierarchy.depths * y_true)
        leaf_pred = np.argmax(self.hierarchy.depths * y_pred)
        return self.cost_matrix[leaf_true, leaf_pred]


    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        return self.brute_force_decode(p_nodes)

    def brute_force_decode(self, p_nodes: np.ndarray) -> np.ndarray:
        all_candidates = self.hierarchy.leaf_events.copy()
        return self._helper_brute_force(p_nodes, all_candidates)
    
    def _load_cost_matrix(self, cost_matrix_path: str) -> np.ndarray:
        """
        Load the cost matrix from a pkl file.
        :param cost_matrix_path: Path to the cost matrix file.
        :return: Cost matrix as a numpy array.
        """
        with open(cost_matrix_path, 'rb') as f:
            cost_matrix = pickle.load(f)
        if not isinstance(cost_matrix, np.ndarray):
            raise ValueError("Cost matrix must be a numpy array.")
        return cost_matrix
    
    def _helper_brute_force(self, p_nodes: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Helper function to brute force decode.
        """
        predictions_opts = np.zeros_like(p_nodes, dtype=int)
        for i in range(p_nodes.shape[0]):
            # Get the indices of the candidates for the current sample
            candidate_indices = np.where(candidates[i])[0]
            if len(candidate_indices) == 0:
                raise ValueError("No candidates found for the current sample. This is not expected.")
            else :
                best_candidate = np.argmin(self.cost_matrix[candidate_indices] @ p_nodes[i, self.hierarchy.leaves_idx])
                predictions_opts[i] = [1 if n in self.hierarchy.get_ancestors(candidate_indices[best_candidate]) else 0 for n in range(self.hierarchy.n_nodes)]
        return predictions_opts
    
    def check_if_y_pred_is_a_single_leaf(self, y_pred):
        """
        Check if the prediction is for a single node.
        This is used to ensure that the node metric can be computed correctly.
        """

        depths_pred = self.hierarchy.depths * y_pred

        return (np.bincount(depths_pred)[1:].max() == 1) & (np.argmax(self.hierarchy.depths * y_pred) in self.hierarchy.leaves_idx)