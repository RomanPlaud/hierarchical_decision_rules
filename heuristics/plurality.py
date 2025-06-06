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
        Decode node-wise predictions to binary vectors using a plurality approach.
        """
        def decode_rec(node, cand, p, maxi):
            if not (node in self.hierarchy_dico.keys()):
                cand[node] = 1
                return cand
            else :
                cand[node] = 1
                children = self.hierarchy_dico[node]
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

        all_candidates = np.zeros_like(proba_nodes)
        for p_i, cand_i in zip(proba_nodes, all_candidates):
            _ = decode_rec(self.hierarchy.root_idx, cand_i, p_i, 0)
        return all_candidates 