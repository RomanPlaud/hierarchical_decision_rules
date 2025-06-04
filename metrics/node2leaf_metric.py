from base_metric import Metric
import numpy as np
import networkx as nx

class Node2LeafMetric(Metric):
    def __init__(self, hierarchy, cost_matrix):
        super().__init__(hierarchy)
        self.cost_matrix = cost_matrix
        self.self._check_hierarchical_constraints()
    
    def metric(self, y_true, y_pred):
        if not self.check_if_y_pred_is_a_single_node(y_pred):
            raise ValueError("multiple nodes is not supported in Node2LeafMetric")

        leaf_true = np.argmax(self.hierarchy.depths * y_true)
        node_pred = np.argmax(self.hierarchy.depths * y_pred)
        return self.cost_matrix[leaf_true, node_pred]


    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        pass

def _is_strictly_hierarchically_reasonable(self):
    """
    Check if the cost matrix satisfies the hierarchical constraints:
    For each node n, and its parent pi(n):
    - For each leaf l in L(n), C(n, l) < C(pi(n), l)
    - For each leaf l not in L(n), C(n, l) > C(pi(n), l) (Definition 4.2 from the paper)
    """
    

    for n in range(self.hierarchy.n_nodes):
        if n == self.hierarchy.root_idx:
            continue

        pi_n = self.hierarchy.parent[n]

        leaf_descendants = set(self.hierarchy.compute_leaf_descendants_index(n))
        not_leaf_descendants = self.hierarchy.leave_idx - leaf_descendants

        # Convert to list for indexing
        L_n = list(leaf_descendants)
        not_L_n = list(not_leaf_descendants)

        # Check C(n, l) < C(pi(n), l) for l in L(n)
        if L_n:
            c_n = self.cost_matrix[n, L_n]
            c_pi_n = self.cost_matrix[pi_n, L_n]
            if not np.all(c_n < c_pi_n):
                return False

        # Check C(n, l) > C(pi(n), l) for l not in L(n)
        if not_L_n:
            c_n = self.cost_matrix[n, not_L_n]
            c_pi_n = self.cost_matrix[pi_n, not_L_n]
            if not np.all(c_n > c_pi_n):
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
    for n in range(self.hierarchy.n_nodes):
        if n == self.hierarchy.root_idx:
            continue

        pi_n = self.hierarchy.parent[n]

        leaf_descendants = set(self.hierarchy.compute_leaf_descendants_index(n))
        not_leaf_descendants = self.hierarchy.leave_idx - leaf_descendants

        # Convert to list for indexing
        L_n = list(leaf_descendants)
        not_L_n = list(not_leaf_descendants)

        # Check C(n, l) < C(pi(n), l) for l in L(n)
        if L_n:
            c_n = self.cost_matrix[n, L_n]
            c_pi_n = self.cost_matrix[pi_n, L_n]
            if not np.all(c_n < c_pi_n):
                return False

        # check C(n, l) > C(pi(n), l) for l not in L(n) and LCA(l, n) != r
        