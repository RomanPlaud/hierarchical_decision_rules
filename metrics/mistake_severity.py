from .base_metric import Metric
import numpy as np
import networkx as nx

class MistakeSeverity(Metric):
    def __init__(self, hierarchy):
        super().__init__(hierarchy)
        g = hierarchy.hierarchy_graph.to_undirected()
        self.distances = dict(nx.all_pairs_shortest_path_length(g))

    def metric(self, y_true, y_pred):

        if not self.check_if_y_pred_is_a_single_node(y_pred):
            raise ValueError("multiple nodes is not supported in MistakeSeverity metric")

        leaf_true = np.argmax(self.hierarchy.depths * y_true)
        node_pred = np.argmax(self.hierarchy.depths * y_pred)
        return self.distances[leaf_true][node_pred]
    
    def decode(self, p_nodes: np.ndarray) -> np.ndarray:
        """
        Decode node-wise predictions to binary vectors.
        Here, optimal decoding is to threshold the probabilities at 0.5.
        """
        return (p_nodes > 0.5).astype(int)

