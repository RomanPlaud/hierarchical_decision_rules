from .base_metric import Metric
import numpy as np

class hFBetaScore(Metric):
    def __init__(self, hierarchy, beta: float = 1.0):
        """
        Initialize the F-beta score metric.
        :param hierarchy: The hierarchy of labels.
        :param beta: The beta parameter for the F-beta score.
                     beta < 1 favors precision, beta > 1 favors recall.
        """
        super().__init__(hierarchy)
        self.beta = beta
        self.q_min_beta = self._find_constants()

def metric(self, y_true : np.ndarray, y_pred : np.ndarray) -> float:
    tp = np.sum(y_true*y_pred)
    fp = np.sum(y_pred*(1-y_true))
    fn = np.sum((1-y_pred)*y_true)

    pr = tp/(tp+fp)
    re = tp/(tp+fn)
    if pr+re == 0:
        return 0
    else:
        return (1+self.beta**2)*pr*re/((self.beta**2)*pr+re)

def recall(y_true : np.ndarray, y_pred : np.ndarray) -> float:
        tp = np.sum(y_true*y_pred)
        fn = np.sum((1-y_pred)*y_true)
        if tp+fn == 0:
            return 0
        else:
            return tp/(tp+fn)
    
def precision(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    tp = np.sum(y_true*y_pred)
    fp = np.sum(y_pred*(1-y_true))
    if tp+fp == 0:
        return 0
    else:
        return tp/(tp+fp)

def _find_constants(self):
    """
    Computes the constants q_max^C(n)
    """
    q_min_beta = 1 / (1 + (self.beta)**2 * (self.hierarchy.depth_max_descendants + 1))
    return q_min_beta
    

def _compute_delta(self, probas_nodes):
        """
        Compute delta tensor of shape (n_samples, k_max+1, n_nodes), where:
        - k_max = max over samples of (# nodes with prob >= q_min_beta[n]),
        - ∆^β_k(n) = ∑_l∈L(n) p(l) * (1+β²) / [k + β²(d(l)+1)]

        Args:
            probas_nodes (ndarray): shape (n_samples, n_nodes), predicted probabilities.

        Returns:
            delta (ndarray): shape (n_samples, k_max+1, n_nodes)
        """
        k_max = np.max(np.sum(probas_nodes >= self.q_min_beta, axis=1))
        n_samples = probas_nodes.shape[0]
        delta = np.zeros((n_samples, k_max + 1, self.n_nodes))

        def recurse(node, depth, k):
            if node in self.hierarchy.leaves_idx:
                weight = (1 + self.beta ** 2) / (k + (self.beta ** 2) * (depth + 1))
                dval = probas_nodes[:, node] * weight
            else:
                dval = sum(recurse(child, depth + 1, k) for child in self.hierarchy.hierarchy_dico[node])
            delta[:, k, node] = dval
            return dval

        for k in range(1, k_max + 1):
            recurse(self.hierarchy.root_idx, 0, k)

        return delta

def decode(self, probas_nodes):
        """
        Decode the predicted probabilities into a binary matrix of shape (n_samples, n_nodes).
        """
        predictions_opt = np.zeros_like(probas_nodes)
        # all_f = np.zeros(self.hierarchy.n_nodes)
        delta = self._compute_delta(probas_nodes)

        candidate_nodes = (probas_nodes >= self.q_min_beta)

        for i, (cand_i, p_i) in enumerate(zip(candidate_nodes, probas_nodes)):
            idx_cand_i = np.where(cand_i)[0]
            k_max_i = len(idx_cand_i)
            value_pred_max = 0
            for k in range(1, k_max_i + 1):
                prediction_i, value_pred = np.zeros(self.hierarchy.n_nodes), 0
                top_k_nodes_delta = np.argsort(delta[i, k, idx_cand_i])[-k:]
                prediction_i[idx_cand_i[top_k_nodes_delta]] = 1
                value_pred = np.sum(delta[i, k, idx_cand_i[top_k_nodes_delta]])
                if value_pred > value_pred_max:
                    value_pred_max = value_pred
                    predictions_opt[i] = prediction_i
        return predictions_opt