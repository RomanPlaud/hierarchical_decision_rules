import networkx as nx
import numpy as np

def initialize_hierarchy_from_dico(hierarchy_dico):
    # hierarchy_dico is a dictionary whose keys are the names of the classes and values are the names of the children classes of the key
    # initialize a nx graph from the dictionary hierarchy_dico
    hierarchy = nx.DiGraph()
    for key in hierarchy_dico.keys():
        for child in hierarchy_dico[key]:
            hierarchy.add_edge(key, child)
    # check if the hierarchy is a tree
    if not nx.is_tree(hierarchy):
        raise ValueError("The hierarchy is not a tree.")
    return hierarchy

class Hierarchy:
    def __init__(self, hierarchy_dico_idx):

        self._check_format_hierarchy(hierarchy_dico_idx)
        self.hierarchy_graph =   initialize_hierarchy_from_dico(self.hierarchy_dico_idx)

        self.root_idx = self._get_root_idx()
        self.leaf_events = self._get_leaf_events()

        self.parent = self._get_parent_mapping()
        self.depths = self._get_depths()
        self.depth_max_descendants = self._get_max_depth_of_descendants()
        
        # self.parent = None
        # self.depths = None
        # self.max_depth = None
        # self.d_max = None
        # self.non_root_lowest_ancestor = None

    def _check_format_hierarchy(self, hierarchy_dico_idx):
        """
        Check that leaf nodes get the first indices (i.e., index 0 to n_leaves - 1).
        Check that non-leaf nodes get the indices from n_leaves to n_nodes - 1.
        Check that the hierarchy_dico_idx is a valid hierarchy (i.e., no cycles, single root).
        """
        # check hierarchy_dico_idx is a dictionary
        if not isinstance(hierarchy_dico_idx, dict):
            raise TypeError("Hierarchy must be a dictionary mapping node indices to their children indices.")
        # check hierarchy_dico_idx is not empty
        if not hierarchy_dico_idx:
            raise ValueError("Hierarchy cannot be empty.")
        # check that all keys are integers
        if not all(isinstance(k, int) for k in hierarchy_dico_idx.keys()):
            raise TypeError("All keys in hierarchy_dico_idx must be integers.")
        # check that all values are lists of integers
        if not all(isinstance(v, list) and all(isinstance(i, int) for i in v) for v in hierarchy_dico_idx.values()):
            raise TypeError("All values in hierarchy_dico_idx must be lists of integers.")
        # check that all indices are unique
        all_indices = set(hierarchy_dico_idx.keys())
        for children in hierarchy_dico_idx.values():
            all_indices.update(children)
        if len(all_indices) != len(set(all_indices)):
            raise ValueError("Hierarchy indices must be unique.")
        # check that the hierarchy is a tree
        if not nx.is_tree(initialize_hierarchy_from_dico(hierarchy_dico_idx)):
            raise ValueError("Hierarchy must be a tree (i.e., no cycles, single root).")
        # get all leaves
        parents = set(hierarchy_dico_idx.keys())
        all_children = set([child for children in hierarchy_dico_idx.values() for child in children])
        nodes = parents.union(all_children)
        leaves_idx = list(nodes - parents)
        if max(leaves_idx) >= len(leaves_idx):
            raise ValueError("Leaf indices must be in the range [0, n_leaves - 1].")
        if max(nodes) >= len(nodes):
            raise ValueError("Node indices must be in the range [0, n_nodes - 1].")
        self.hierarchy_dico_idx = hierarchy_dico_idx
        self.n_nodes = len(nodes)
        self.n_leaves = len(leaves_idx)
        self.leaves_idx = np.arange(self.n_leaves)
    
    def _get_root_idx(self):
        """
        Identify the single root node (in-degree zero) in the hierarchy_graph_idx.
        """
        # use nx to find the root in the self.hierarchy_graph
        root_idx = [n for n in self.hierarchy_graph.nodes() if self.hierarchy_graph.in_degree(n) == 0][0]
        return root_idx

    def _get_leaf_events(self):
        """
        Build an event-encoding matrix of shape (n_leaves, n_nodes),
        where each row corresponds to one leaf, and entries are 1 if the node
        lies on the path from root to that leaf, else 0.
        """
        leaf_events = np.zeros((self.n_leaves, self.n_nodes), dtype=int)
        def dfs(node, path):
            path = path + [node]
            if node in self.leaves_idx:
                for ancestor in path:
                    # node is in leaves_idx, and we ensured leaves have indices 0..n_leaves-1
                    leaf_events[node, ancestor] = 1
            else:
                for child in self.hierarchy_dico_idx[node]:
                    dfs(child, path)

        dfs(self.root_idx, [])
        return leaf_events
        
    def _get_parent_mapping(self):
        """
        Build a map self.parent: child -> parent for every non-root node.
        """
        parents = {}
        for parent, children in self.hierarchy_dico_idx.items():
            for child in children:
                parents[child] = parent

        assert self.root_idx not in parents, "Root node should not have a parent."
        return parents


    def _get_depths(self):
        """
        Populate depths: array of depth for every node.
        """
        depths = np.zeros(self.n_nodes, dtype=int)

        def recurse(node, depth):
            depths[node] = depth
            if node in self.leaves_idx:
                return
            else:
                for child in self.hierarchy_dico_idx[node]:
                    recurse(child, depth + 1)

        recurse(self.root_idx, 0)
        return depths

    def _get_max_depth_of_descendants(self):
        """
        Populate depth_max_descendants: array of maximum depth of any leaf descendant of node.
        """
        depth_max_descendants = np.zeros(self.n_nodes, dtype=int)

        def recurse(node):
            if node in self.leaves_idx:
                depth_max_descendants[node] = self.depths[node]
                return self.depths[node]
            max_child_depth = np.max([recurse(child) for child in self.hierarchy_dico_idx[node]])
            depth_max_descendants[node] = max_child_depth
            return max_child_depth

        recurse(self.root_idx)
        self.depth_max = depth_max_descendants[self.root_idx] 
        return depth_max_descendants


    def compute_leaf_descendants_index(self, node: int) -> list[int]:
        """
        Recursively return a list of leaf node indices descending from the given node.
        """
        if node in self.leaves_idx:
            return [node]
        descendants = []
        for child in self.hierarchy_dico_idx.get(node, []):
            descendants.extend(self.compute_leaf_descendants_index(child))
        return descendants

    def get_probas(self, probas_leaves):
        """
        Given an array probas_leaves of shape (n_samples, n_leaves),
        compute self.probas_nodes of shape (n_samples, n_nodes),
        where each node's probability is the sum of its leaf descendants'.
        """
        if probas_leaves.shape[1] != self.n_leaves:
            raise ValueError(
                f"Probas shape mismatch: got {probas_leaves.shape[1]}, "
                f"expected number of columns = {self.n_leaves}"
            )

        n_samples = probas_leaves.shape[0]
        self.probas_nodes = np.zeros((n_samples, self.n_nodes))

        def assign_proba_rec(node):
            if node in self.leaves_idx:
                node_prob = probas_leaves[:, self.leaf2i[node]]
                self.probas_nodes[:, node] = node_prob
                return node_prob
            children_sum = sum(assign_proba_rec(child) for child in self.hierarchy_dico[node])
            self.probas_nodes[:, node] = children_sum
            return children_sum

        assign_proba_rec(self.root)
        return self.probas_nodes


    def get_ancestors(self, node):
        """
        Return a list of ancestor node IDs for the given node,
        starting from the node itself up to the root.
        """
        ancestors = []
        current = node
        while current != self.root_idx:
            ancestors.append(current)
            current = self.parent[current]
        ancestors.append(self.root_idx)
        return ancestors

    def compute_non_root_lowest_ancestor(self):
        """
        For every node (except root), record the first child of root
        on its path from the root. The root maps to itself.
        """
        self.ensure_structure()
        self.non_root_lowest_ancestor = {self.root: self.root}

        def recurse(lowest_ancestor, node):
            self.non_root_lowest_ancestor[node] = lowest_ancestor
            for child in self.hierarchy_dico.get(node, []):
                recurse(lowest_ancestor, child)

        for child in self.hierarchy_dico.get(self.root, []):
            recurse(child, child)

    def compute_information_content(self):
        """
        Compute self.information_content[node] = log2(N_leaves / n_descendants(node))
        for each node, where n_descendants is the count of leaf descendants.
        """
        self.ensure_structure()
        n_leaves = len(self.leaves)
        self.information_content = np.zeros(len(self.hierarchy_graph))

        def recurse(node):
            if node in self.leaves:
                count = 1
                self.information_content[node] = np.log2(n_leaves)
                return count
            count = sum(recurse(child) for child in self.hierarchy_dico[node])
            self.information_content[node] = np.log2(n_leaves / count)
            return count

        recurse(self.root)
        return self.information_content

    def compute_delta_general(self, probas_leaves, beta=1):
        """
        Compute delta tensor of shape (n_samples, k_max+1, n_nodes), where
        k_max = max over samples of (# nodes with prob >= p_max),
        p_max = 1 / [1 + beta^2 * (max_depth + 1)].
        """
        self.ensure_structure()
        self.compute_depth_max()
        self.get_probas(probas_leaves)

        denom = 1 + (beta ** 2) * (self.max_depth + 1)
        p_max = 1 / denom
        # For each sample, count nodes with prob >= p_max; take the maximum
        k_max = int(np.max(np.sum(self.probas_nodes >= p_max, axis=1)))

        n_samples, n_nodes = self.probas_nodes.shape
        delta = np.zeros((n_samples, k_max + 1, n_nodes))

        def recurse(node, depth, k):
            if node in self.leaves:
                weight = (1 + beta ** 2) / (k + (beta ** 2) * (depth + 1))
                dval = self.probas_nodes[:, node] * weight
            else:
                dval = sum(recurse(child, depth + 1, k) for child in self.hierarchy_dico[node])
            delta[:, k, node] = dval
            return dval

        for k in range(1, k_max + 1):
            recurse(self.root, 0, k)

        return delta

    def compute_delta_beta(self, probas_leaves, beta=1):
        """
        Wrapper around compute_delta_general, with an explicit beta parameter.
        """
        return self.compute_delta_general(probas_leaves, beta=beta)