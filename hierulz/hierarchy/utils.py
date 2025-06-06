import networkx as nx

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