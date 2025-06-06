from ..metrics.leaf2leaf_metric import Leaf2LeafMetric
from .base_heuristic import Heuristic
import numpy as np

class CRM_BM(Heuristic):
    def __init__(self, hierarchy):
        """
        Initialize the CRM_BM heuristic with a hierarchy object and a metric.
        """
        super().__init__(hierarchy)
        cost_matrix = self.get_distance_matrix()

        self.metric = Leaf2LeafMetric(hierarchy, cost_matrix)

    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors using the CRM_BM heuristic.
        """
        full_predictions = np.zeros_like(proba_nodes)
        for i, p_i in enumerate(proba_nodes):
            if np.max(p_i) >= 0.5:
                # Get the index of the maximum probability
                node_predicted = np.argmax(p_i)
                # Get the leaf nodes that are ancestors of the max_idx
                ancestors = self.hierarchy.leaf_events[node_predicted]
                # Set the corresponding indices in full_predictions to 1
                full_predictions[i, ancestors] = 1
            else:
                full_predictions[[i]] = self.metric.decode(p_i[None])
        return full_predictions

    def get_distance_matrix(self):
        """
        Get the distance matrix for all pairs of leaf nodes in the hierarchy.
        """
        cost_matrix = np.zeros((self.hierarchy.n_leaves, self.hierarchy.n_leaves))
        for i, leaf1 in enumerate(self.hierarchy.leaves_idx):
            for j, leaf2 in enumerate(self.hierarchy.leaves_idx):
                if i != j:
                    dist_ij = nx.shortest_path_length(
                        self.hierarchy.hierarchy_graph, source=leaf1, target=leaf2)
                    cost_matrix[i, j], cost_matrix[j, i] = dist_ij, dist_ij
        return cost_matrix
                    