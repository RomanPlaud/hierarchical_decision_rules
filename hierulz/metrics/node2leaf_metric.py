from .base_metric import Metric
import numpy as np
import pickle
import sys
infty = sys.float_info.max

class Node2LeafMetric(Metric):
    def __init__(self, hierarchy, cost_matrix):
        super().__init__(hierarchy)
        if isinstance(cost_matrix, str):
            self.cost_matrix = self._load_cost_matrix(cost_matrix)
        elif isinstance(cost_matrix, np.ndarray):
            self.cost_matrix = cost_matrix        
        self._check_hierarchical_constraints()
        self._find_constants()
    
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
        See Theorem 4.4 from the paper
        """
        all_candidates = np.ones_like(p_nodes, dtype=bool)

        for n in range(self.hierarchy.n_nodes):
            if n == self.hierarchy.root_idx:
                continue
            
            pi_n = self.hierarchy.parent[n]
            q_min_n = self.q_min[n]
            q_max_n = self.q_max[n]

            # delete node n if proba of n is strictly less than self.q_min(n)
            all_candidates[:, n] &= (p_nodes[:, n] >= q_min_n)

            # delete parent node of node n if proba of n is strictly more than self.q_max(n)
            all_candidates[:, pi_n] &= (p_nodes[:, n] <= q_max_n)

        return self._helper_brute_force(p_nodes, all_candidates)
    
    def decode_softly_reasonable(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to leaf nodes based on softly reasonable cost matrix.
        See Proposition E.7 from the paper
        """
        all_candidates = np.ones_like(p_nodes, dtype=bool)
        # for each node n, compute the proba of its shallowest common ancestor, use self.hierarchy.non_root_lowest_ancestor
        # to compute the shallowest common ancestor
        p_shallowest_common_ancestor = p_nodes[:, self.hierarchy.non_root_lowest_ancestor]


        for n in range(self.hierarchy.n_nodes):
            if n == self.hierarchy.root_idx:
                continue
            
            pi_n = self.hierarchy.parent[n]
            q_min_n = self.q_min[n]
            q_max_n = self.q_max[n]

            # delete node n if proba of n is strictly less than self.q_min(n)
            all_candidates[:, n] &= (p_nodes[:, n] >= (q_min_n * p_shallowest_common_ancestor[:, n]) )

            # delete parent node of node n if proba of n is strictly more than self.q_max(n)
            all_candidates[:, pi_n] &= (p_nodes[:, n] <= (q_max_n * p_shallowest_common_ancestor[:, n]) )

        return self._helper_brute_force(p_nodes, all_candidates)
    
    def brute_force_decode(self, p_nodes: np.ndarray) -> np.ndarray:
        all_candidates = np.ones_like(p_nodes, dtype=bool)
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
         # check if can be transposed to match the dimensions of the hierarchy
        if cost_matrix.shape[0] == self.hierarchy.n_leaves or cost_matrix.shape[1] != self.hierarchy.n_nodes:
            cost_matrix = cost_matrix.T
        if cost_matrix.shape[0] != self.hierarchy.n_nodes or cost_matrix.shape[1] != self.hierarchy.n_leaves:
            raise ValueError(f"Cost matrix shape {cost_matrix.shape} does not match hierarchy dimensions: {self.hierarchy.n_nodes} nodes and {self.hierarchy.n_leaves} leaves.")
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
                    ancestors_n = self.hierarchy.get_ancestors(n)
                    # check fot size of overlap between ancestors of n and leaf_idx
                    if len(np.intersect1d(ancestors_n, idx_leaf_event)) > 1:
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
        Computes the constants q_min^C(n) and q_max^C(n)
        """
        if self.is_strictly_hierarchically_reasonable:
            self._find_constants_strictly_reasonable()
        elif self.is_softly_hierarchically_reasonable:
            self._find_constants_softly_reasonable()

    def _find_constants_strictly_reasonable(self):
        """
        Computes the constants q_min^C(n) and q_max^C(n) for each node n in the hierarchy
        for strictly reasonable cost matrix.
        Check Lemma (4.3) from the paper
        """
        Mn_bar = infty*np.ones(self.hierarchy.n_nodes)
        Mn = infty*np.ones(self.hierarchy.n_nodes)
        mn = np.zeros(self.hierarchy.n_nodes)
        mn_bar = np.zeros(self.hierarchy.n_nodes)

        for leaf_idx, leaf_event in enumerate(self.hierarchy.leaf_events):
            idx_leaf_event = np.where(leaf_event)[0]

            for n in range(self.hierarchy.n_nodes):
                if n == self.hierarchy.root_idx or (n in self.hierarchy.hierarchy_dico_idx[self.hierarchy.root_idx]):
                    continue
                pi_n = self.hierarchy.parent[n]
                if n in idx_leaf_event:
                    Mn[n] = max(Mn[n], (self.cost_matrix[pi_n, leaf_idx] - self.cost_matrix[n, leaf_idx]))
                    mn[n] = min(mn[n], (self.cost_matrix[pi_n, leaf_idx] - self.cost_matrix[n, leaf_idx]))
                else:
                    Mn_bar[n] = max(Mn_bar[n], (self.cost_matrix[n, leaf_idx] - self.cost_matrix[pi_n, leaf_idx]))
                    mn_bar[n] = min(mn_bar[n], (self.cost_matrix[n, leaf_idx] - self.cost_matrix[pi_n, leaf_idx]))
                
        self.q_max = np.divide(Mn_bar, Mn_bar + mn, out=np.ones_like(Mn_bar), where=(Mn_bar + mn) != 0)
        self.q_min = np.divide(mn_bar, mn_bar + Mn, out=np.zeros_like(mn_bar), where=(mn_bar + Mn) != 0)


    def _find_constants_softly_reasonable(self):
        """
        Computes the constants q_min^C(n) and q_max^C(n) for each node n in the hierarchy
        for strictly reasonable cost matrix.
        Check Proposition E.6 (Appendix) from the paper
        """
        Mn_bar = infty*np.ones(self.hierarchy.n_nodes)
        Mn = infty*np.ones(self.hierarchy.n_nodes)
        mn = np.zeros(self.hierarchy.n_nodes)
        mn_bar = np.zeros(self.hierarchy.n_nodes)

        for leaf_idx, leaf_event in enumerate(self.hierarchy.leaf_events):
            idx_leaf_event = np.where(leaf_event)[0]

            for n in range(self.hierarchy.n_nodes):
                if n == self.hierarchy.root_idx or (n in self.hierarchy.hierarchy_dico_idx[self.hierarchy.root_idx]):
                    continue
                pi_n = self.hierarchy.parent[n]
                if n in idx_leaf_event:
                    Mn[n] = max(Mn[n], (self.cost_matrix[pi_n, leaf_idx] - self.cost_matrix[n, leaf_idx]))
                    mn[n] = min(mn[n], (self.cost_matrix[pi_n, leaf_idx] - self.cost_matrix[n, leaf_idx]))
                else:
                    if len(np.intersect1d(self.hierarchy.get_ancestors(n), idx_leaf_event)) > 1:
                        Mn_bar[n] = max(Mn_bar[n], (self.cost_matrix[n, leaf_idx] - self.cost_matrix[pi_n, leaf_idx]))
                        mn_bar[n] = min(mn_bar[n], (self.cost_matrix[n, leaf_idx] - self.cost_matrix[pi_n, leaf_idx]))
                
        self.q_max = np.divide(Mn_bar, Mn_bar + mn, out=np.ones_like(Mn_bar), where=(Mn_bar + mn) != 0)
        self.q_min = np.divide(mn_bar, mn_bar + Mn, out=np.zeros_like(mn_bar), where=(mn_bar + Mn) != 0)


