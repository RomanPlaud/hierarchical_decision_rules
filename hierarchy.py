import networkx as nx
import numpy as np

def initialize_hierarchy_from_dico(hierarchy_dico):
    # hierarchy_dico is a dictionary whose keys are the names of the classes and values are the names of the children classes of the key
    # initialize a nx graph from the dictionary hierarchy_dico
    hierarchy = nx.DiGraph()
    for key in hierarchy_dico.keys():
        for child in hierarchy_dico[key]:
            hierarchy.add_edge(key, child)
    return hierarchy

class Hierarchy:
    def __init__(self, hierarchy_dico=None):
        self.hierarchy_dico = hierarchy_dico
        self.hierarchy_graph = (
            initialize_hierarchy_from_dico(hierarchy_dico)
            if hierarchy_dico is not None else None
        )
        self.leaves_idx = None
        self.i2leaf = None
        self.leaf2i = None
        self.root = None
        self.parent = None
        self.depths = None
        self.max_depth = None
        self.probas_nodes = None
        self.events = None
        self.information_content = None
        self.d_max = None
        self.non_root_lowest_ancestor = None

    def create_binary_hierarchy(self, size):
        """
        Build a full binary tree of given size (number of leaves).
        """
        self.hierarchy_graph = nx.full_rary_tree(2, size, create_using=nx.DiGraph)
        # Build hierarchy_dico from graph successors, filtering out leaves
        self.hierarchy_dico = {
            node: list(self.hierarchy_graph.successors(node))
            for node in self.hierarchy_graph.nodes()
            if self.hierarchy_graph.out_degree(node) > 0
        }

    def ensure_structure(self):
        """
        Ensure that leaves and root have been identified.
        """
        if self.leaves is None:
            self.get_leaves()
        if self.root is None:
            self.get_root()

    def get_leaves(self):
        """
        Populate:
          - self.leaves: list of leaf-node IDs
          - self.i2leaf: map from index to leaf ID
          - self.leaf2i: map from leaf ID to index
        """
        self.leaves = [
            node
            for node in self.hierarchy_graph.nodes()
            if self.hierarchy_graph.out_degree(node) == 0
        ]
        self.i2leaf = {i: leaf for i, leaf in enumerate(self.leaves)}
        self.leaf2i = {leaf: i for i, leaf in enumerate(self.leaves)}

    def get_root(self):
        """
        Identify the single root node (in-degree zero).
        """
        self.root = next(node for node, deg in self.hierarchy_graph.in_degree() if deg == 0)

    def compute_parent(self):
        """
        Build a map self.parent: child -> parent for every non-root node.
        """
        self.parent = {
            child: parent
            for parent, children in self.hierarchy_dico.items()
            for child in children
        }

    def compute_depth(self):
        """
        Populate self.depths: array of depth for every node.
        """
        self.ensure_structure()
        self.depths = np.zeros(len(self.hierarchy_graph), dtype=int)

        def recurse(node, depth):
            self.depths[node] = depth
            for child in self.hierarchy_dico.get(node, []):
                recurse(child, depth + 1)

        recurse(self.root, 0)

    def compute_all_d_max(self):
        """
        Populate self.d_max: dictionary node -> maximum depth of any leaf descendant.
        """
        self.ensure_structure()
        if self.depths is None:
            self.compute_depth()

        self.d_max = {}

        def recurse(node):
            if node in self.leaves:
                self.d_max[node] = self.depths[node]
                return self.depths[node]
            max_child_depth = max(recurse(child) for child in self.hierarchy_dico[node])
            self.d_max[node] = max_child_depth
            return max_child_depth

        recurse(self.root)
        self.max_depth = self.d_max[self.root] 
        return self.d_max


    def compute_leaf_descendants_index(self, node):
        """
        Return a list of leaf IDs descending from the given node.
        """
        if node in self.leaves:
            return [node]
        descendants = []
        for child in self.hierarchy_dico[node]:
            descendants.extend(self.compute_leaf_descendants_index(child))
        return descendants

    def get_probas(self, probas_leaves):
        """
        Given an array probas_leaves of shape (n_samples, n_leaves),
        compute self.probas_nodes of shape (n_samples, n_nodes),
        where each node's probability is the sum of its leaf descendants'.
        """
        self.ensure_structure()
        if probas_leaves.shape[1] != len(self.leaves):
            raise ValueError(
                f"Probas shape mismatch: got {probas_leaves.shape}, "
                f"expected number of columns = {len(self.leaves)}"
            )

        n_samples = probas_leaves.shape[0]
        n_nodes = len(self.hierarchy_graph)
        self.probas_nodes = np.zeros((n_samples, n_nodes))

        def assign_proba_rec(node):
            if node in self.leaves:
                node_prob = probas_leaves[:, self.leaf2i[node]]
                self.probas_nodes[:, node] = node_prob
                return node_prob
            children_sum = sum(assign_proba_rec(child) for child in self.hierarchy_dico[node])
            self.probas_nodes[:, node] = children_sum
            return children_sum

        assign_proba_rec(self.root)
        return self.probas_nodes

    def get_events(self):
        """
        Build an event-encoding matrix of shape (n_leaves, n_nodes),
        where each row corresponds to one leaf, and entries are 1 if the node
        lies on the path from root to that leaf, else 0.
        """
        self.ensure_structure()
        events_list = []
        for leaf in self.leaves:
            ancestors = nx.shortest_path(self.hierarchy_graph, self.root, leaf)
            event_vector = np.array(
                [1 if node in ancestors else 0 for node in range(len(self.hierarchy_graph))]
            )
            events_list.append(event_vector)
        self.events = np.vstack(events_list)

    def get_ancestors(self, node):
        """
        Return a list of ancestor node IDs for the given node,
        starting from the node itself up to the root.
        """
        self.compute_parent()
        ancestors = []
        current = node
        while current != self.root:
            ancestors.append(current)
            current = self.parent[current]
        ancestors.append(self.root)
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