from .base_metric import Metric
import numpy as np
import networkx as nx
from tqdm import tqdm

class Node2LeafMetric(Metric):
    def __init__(self, hierarchy, cost_matrix):
        super().__init__(hierarchy)
        self.cost_matrix = cost_matrix
        self._check_hierarchical_constraints()
    
    def metric(self, y_true, y_pred):
        if not self.check_if_y_pred_is_a_single_node(y_pred):
            raise ValueError("multiple nodes is not supported in Node2LeafMetric")

        leaf_true = np.argmax(self.hierarchy.depths * y_true)
        node_pred = np.argmax(self.hierarchy.depths * y_pred)
        return self.cost_matrix[leaf_true, node_pred]


    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        if self.is_strictly_hierarchically_reasonable:
            return self.decode_strictly_reasonable(p_nodes)
        elif self.is_softly_hierarchically_reasonable:
            return self.decode_softly_reasonable(p_nodes)
        else:
            return self.brute_force_decode(p_nodes)

    def decode_strictly_reasonable(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to leaf nodes based on strictly reasonable cost matrix.
        """
        pass

    def decode_softly_reasonable(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to leaf nodes based on softly reasonable cost matrix.
        """
        pass

    def brute_force_decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to leaf nodes using brute force.
        """
        pass

    def _is_strictly_hierarchically_reasonable(self):
        """
        Check if the cost matrix satisfies the hierarchical constraints:
        For each node n, and its parent pi(n):
        - For each leaf l in L(n), C(n, l) < C(pi(n), l)
        - For each leaf l not in L(n), C(n, l) > C(pi(n), l) (Definition 4.2 from the paper)
        """
        
        for leaf_idx, leaf_event in enumerate(self.hierarchy.leaf_events):
            idx_leaf_event = np.where(leaf_event)[0]

            for n in range(self.hierarchy.n_nodes):
                if n == self.hierarchy.root_idx:
                    continue

                pi_n = self.hierarchy.parent[n]

                if n in idx_leaf_event:
                    if self.cost_matrix[n, leaf_idx] >= self.cost_matrix[pi_n, leaf_idx]:
                        return False
                else:
                    if self.cost_matrix[n, leaf_idx] <= self.cost_matrix[pi_n, leaf_idx]:
                        return False
        return True


    def _is_softly_hierarchically_reasonable(self):
        """
        Check if the cost matrix satisfies the soft hierarchical constraints:
        For each node n, and its parent pi(n):
        - For each leaf l in L(n), C(n, l) < C(pi(n), l)
        - For each leaf l not in L(n), 
            - C(n, l) > C(π(n), l) if LCA(l, n) ̸= r,
            - C(n, l) = C(π(n), l) if LCA(l, n) = r.
        """
        for leaf_idx, leaf_event in tqdm(enumerate(self.hierarchy.leaf_events)):
            idx_leaf_event = np.where(leaf_event)[0]

            for n in range(self.hierarchy.n_nodes):
                if n == self.hierarchy.root_idx:
                    continue

                pi_n = self.hierarchy.parent[n]

                if n in idx_leaf_event:
                    if self.cost_matrix[n, leaf_idx] >= self.cost_matrix[pi_n, leaf_idx]:
                        return False
                else:
                    if nx.lowest_common_ancestor(self.hierarchy.hierarchy_graph, n, leaf_idx) != self.hierarchy.root_idx:
                        if self.cost_matrix[n, leaf_idx] <= self.cost_matrix[pi_n, leaf_idx]:
                            return False
                    else:
                        if self.cost_matrix[n, leaf_idx] != self.cost_matrix[pi_n, leaf_idx]:
                            return False
        return True
            
    def _check_hierarchical_constraints(self):
        self.is_strictly_hierarchically_reasonable = self._is_strictly_hierarchically_reasonable()
        self.is_softly_hierarchically_reasonable = self._is_softly_hierarchically_reasonable()

    def _find_constants(self):
        """
        a
        """
        pass