from .base_heuristic import Heuristic
import numpy as np

class Plurality(Heuristic):
    def __init__(self, hierarchy):
        """
        Initialize the Plurality heuristic with a hierarchy object.
        """
        super().__init__(hierarchy)
        
    def decode(self, proba_nodes: np.ndarray) -> np.ndarray:
        """
        Given node-wise probabilities, select the most probable node 
        whose probability is greater than the maximum probability of all of its non-ancestor node.
        """
        def decode_rec(node, cand, p, maxi):
            if not (node in self.hierarchy.hierarchy_dico_idx.keys()):
                cand[node] = 1
                return cand
            else :
                cand[node] = 1
                children = self.hierarchy.hierarchy_dico_idx[node]
                if len(children) == 1:
                    maxi_node = children[0]
                    decode_rec(maxi_node, cand, p, maxi)
                # find index of the 2 maximum values in p[children]
                else :
                    idx = np.argsort(p[children])[-2:]
                    maxi = max(maxi, p[children][idx[0]])
                    if p[children][idx[1]]>= maxi:
                        maxi_node = children[idx[1]]
                        decode_rec(maxi_node, cand, p, maxi)
                    else: 
                        return cand

        full_predictions = np.zeros_like(proba_nodes)
        for p_i, cand_i in zip(proba_nodes, full_predictions):
            _ = decode_rec(self.hierarchy.root_idx, cand_i, p_i, 0)
        return full_predictions 