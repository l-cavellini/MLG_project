

class Graph():
    def __init__(self, num_nodes, num_relations, num_types, edge_index, edge_type, node_type, features):
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_types = num_types
        self.edge_index = edge_index
        self.edge_type = edge_type
        self.node_type = node_type
        self.features = features
        