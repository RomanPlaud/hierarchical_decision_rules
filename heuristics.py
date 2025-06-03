import numpy as np
import networkx as nx

class DecodingTopDown:
    def __init__(self, hierarchy_dico):
        self.hierarchy_dico = hierarchy_dico

    def decode(self, p_nodes : np.ndarray, root) -> np.ndarray:
        
        def decode_rec(node, h, p):
            if not (node in self.hierarchy_dico.keys()):
                h[node] = 1
                return h
            else :
                h[node] = 1
                children = self.hierarchy_dico[node]
                maxi_node = children[np.argmax(p[children])]
                decode_rec(maxi_node, h, p)

        all_h = np.zeros_like(p_nodes)
        for i, (p, h) in enumerate(zip(p_nodes, all_h)):
            h = decode_rec(root, h, p)
        return all_h 

class PluralityInference:
    def __init__(self, hierarchy_dico):
        self.hierarchy_dico = hierarchy_dico

    def decode(self, p_nodes : np.ndarray, root) -> np.ndarray:
        
        def decode_rec(node, h, p, maxi):
            if not (node in self.hierarchy_dico.keys()):
                h[node] = 1
                return h
            else :
                h[node] = 1
                children = self.hierarchy_dico[node]
                if len(children) == 1:
                    maxi_node = children[0]
                    decode_rec(maxi_node, h, p, maxi)
                # find index of the 2 maximum values in p[children]
                else :
                    idx = np.argsort(p[children])[-2:]
                    maxi = max(maxi, p[children][idx[0]])
                    if p[children][idx[1]]>= maxi:
                        maxi_node = children[idx[1]]
                        decode_rec(maxi_node, h, p, maxi)
                    else: 
                        return h

        all_h = np.zeros_like(p_nodes)
        for i, (p, h) in enumerate(zip(p_nodes, all_h)):
            h = decode_rec(root, h, p, 0)
        return all_h 

class ExpectedInformation:
    def __init__(self, hierarchy_graph, _lambda=0):
        self.hierarchy_graph = hierarchy_graph
        self._lambda = _lambda
    
    def decode(self, p_nodes, information_content, root):
        # max of p_node * information_content
        nodes = np.argmax(p_nodes*(information_content + self._lambda), axis=1)
        # get ancestors of the nodes
        ancestors = [nx.shortest_path(self.hierarchy_graph, root, node) for node in nodes]
        # one hot encoding of the ancestors
        return np.array([[1 if i in ancestors_i else 0 for i in range(len(self.hierarchy_graph))] for ancestors_i in ancestors])
    
    def find_opt(self, probas_val, labels_val, metric, information_content, root, minimize=True):
        best_lambda = 0
        best_score = np.inf if minimize else -np.inf
        for _lambda in np.linspace(0, 1, 10):
            self._lambda = _lambda
            y_pred = self.decode(probas_val, information_content, root)
            score = np.mean([metric(labels_val_i, y_pred_i) for labels_val_i, y_pred_i in zip(labels_val, y_pred)])
            if minimize and score < best_score:
                best_score = score
                best_lambda = _lambda
            elif not minimize and score > best_score:
                best_score = score
                best_lambda = _lambda

        return best_lambda
        
class ConfidenceThreshold:
    def __init__(self, threshold, hierarchy):
        self.threshold = threshold
        self.hierarchy = hierarchy

    def decode(self, p_nodes : np.ndarray, information_content, root) -> np.ndarray:
        idx = p_nodes>self.threshold
        nodes = np.argmax(idx*information_content, axis=1)
        # get ancestors of the nodes
        ancestors = [self.hierarchy.get_ancestors(node) for node in nodes]
        # one hot encoding of the ancestors
        return np.array([[1 if i in ancestors_i else 0 for i in range(len(self.hierarchy_graph))] for ancestors_i in ancestors])

class InformationThreshold:
    def __init__(self, threshold):
        self.threshold = threshold

    def decode(self, p_nodes : np.ndarray, information_content, root) -> np.ndarray:
        idx = information_content>self.threshold
        nodes = np.argmax(idx*p_nodes, axis=1)
        # get ancestors of the nodes
        ancestors = [nx.shortest_path(self.hierarchy_graph, root, node) for node in nodes]
        # one hot encoding of the ancestors
        return np.array([[1 if i in ancestors_i else 0 for i in range(len(self.hierarchy_graph))] for ancestors_i in ancestors])

class CRM_BM:
    def __init__(self, hierarchy):
        hierarchy_graph = hierarchy.hierarchy_graph
        g = hierarchy_graph.to_undirected()
        distances = dict(nx.all_pairs_shortest_path_length(g))
        leaf2i = hierarchy.leaf2i.values()
        self.cost_matrix = np.array([[distances[i][j] for j in leaf2i] for i in leaf2i])
        self.events = hierarchy.generate_events()

    def decode(self, p_events : np.ndarray, root):
        risks = np.dot(self.cost_matrix, p_events.T)
        leaf_preds = np.argmin(risks, axis=0)
        return self.events[leaf_preds]

class HiE:
    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.hierarchy.compute_parent()
        self.leaf_parents = np.array([self.hierarchy.parent[leaf] for leaf in self.hierarchy.leaves])
        self.events = self.hierarchy.generate_events()
    
    def decode(self, p_nodes):
        p_events = p_nodes[:, self.hierarchy.leaves]
        p_parents = p_nodes[:, self.leaf_parents]

        new_p_events = p_events * p_parents 
        preds = np.argmax(new_p_events, axis=1)

        return self.events[preds]
        
class DecodingThreshold:
    def __init__(self, threshold, hierarchy=None):
        self.threshold = threshold
        self.hierarchy = hierarchy
        self.hierarchy.compute_parent()
        self.hierarchy.compute_depth()
        import time

    def decode(self, p_nodes : np.ndarray) -> np.ndarray:
        if self.threshold >= 0.5:
            return np.where(p_nodes >= self.threshold, 1, 0)
        else:
            p_candidates = np.where(p_nodes >= self.threshold, 1, 0)
            # take candidates with highest depth
            # pred = np.array([np.argmax(p_candidates[i]*self.hierarchy.depths) for i in range(len(p_candidates))])
            pred = np.argmax(p_candidates*self.hierarchy.depths, axis=1)
            # return all ancestors of the selected nodes
            # compute time to find res

            return np.array([[1 if i in self.hierarchy.get_ancestors(pred_i) else 0 for i in range(p_nodes.shape[1])] for pred_i in pred])

    # def decode(self, p_nodes : np.ndarray) -> np.ndarray:
    #     return np.where(p_nodes >= self.threshold, 1, 0)
