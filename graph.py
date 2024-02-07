import torch
from torch_geometric.datasets import OGB_MAG
import numpy as np
import pandas as pd
import pickle

class Graph():
    def __init__(self, features, feature_dims, relations, adj_lists, num_nodes, edge_features, edge_dict):
        self.num_nodes = num_nodes
        self.relations = relations
        self.node_types = feature_dims.keys()
        self.num_node_types = len(self.node_types)
        self.adj_lists = adj_lists
        self.features = features
        self.feature_dims = feature_dims
        self.edge_features = edge_features
        self.edge_dict = edge_dict
    
    def get_features(self, nodes, mode):
        return self.features(mode, nodes)
    
    def get_neighbors(self, nodes, mode):
        return self.adj_lists[(nodes, mode)]
    
    def get_edge_index(self, mode):
        return self.adj_lists[mode]

    def get_edge_type(self, mode):
        return self.edge_types[mode]
    

def load_graph(data_dir, embed_dim):
    ''''
    adj_lists = dict of adjacency lists for each relation
    node_maps = dict for each node type
    rels = dict of relations
    '''
    rels, adj_lists, node_maps = pickle.load(open(data_dir+"/graph_data.pkl", "rb"))
    
    node_types_counts = {mode: len(node_maps[mode]) for mode in node_maps} # count the number of nodes for each relation
    num_nodes = sum(node_types_counts.values()) # total number of nodes
    new_node_maps = torch.ones(num_nodes + 1, dtype=torch.long).fill_(-1)
    # a relation is <node, edge, node> and I want a unique list of edges
    
    temp = []
    for each in adj_lists.keys():
        temp.append(each[1])
    edge_types = np.unique(temp) # create a list of unique edges
    
    # create a dictionary of embeddings for each edge type
    edge_features =  torch.nn.Embedding(len(edge_types), embed_dim)  
    edge_features.weight.data.normal_(0, 1./embed_dim)
    edge_dict = {edge: i for i, edge in enumerate(edge_types)}

    for t, id_list in node_maps.items():
        for i, n in enumerate(id_list):
            assert new_node_maps[n] == -1
            new_node_maps[n] = i

    node_maps = new_node_maps
    feature_dims = {r : embed_dim for r in rels}
    feature_modules = {m : torch.nn.Embedding(node_types_counts[m] + 1, embed_dim) for m in rels} # create a dictionary of embeddings for each relation
    for mode in rels:
        feature_modules[mode].weight.data.normal_(0, 1./embed_dim)
    
    features = lambda nodes, mode: feature_modules[mode](node_maps[nodes])

    graph = Graph(features, feature_dims, rels, adj_lists, num_nodes, edge_features, edge_dict)
    return graph, feature_modules, node_maps
