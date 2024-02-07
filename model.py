import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
import math


class HGT_Layer(MessagePassing):
    def __init__(self, n_head, in_dim, out_dim, edge_types, node_types, dropout = 0.2, **kwargs):
        super(HGT_Layer, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.n_head = n_head
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.node_types = {node_type: i for i, node_type in enumerate(node_types)}
        self.num_types = len(node_types)
        self.edge_types = {edge_types: i for i, edge_types in enumerate(edge_types)}
        self.num_edge_types = len(edge_types)
        self.k_dim = out_dim // n_head
        self.sqrt_k = math.sqrt(self.k_dim)
        self.att = None

        self.k_linear = nn.ModuleList()
        self.q_linear = nn.ModuleList()
        self.v_linear = nn.ModuleList()
        self.a_linear = nn.ModuleList()

        self.relation_prior = nn.Parameter(torch.ones(self.num_edge_types, n_head)) # prior for the meta-relations
        self.relation_att = nn.Parameter(torch.Tensor(self.num_edge_types, n_head, self.k_dim, self.k_dim)) # for each edge_type and for each head, we have a relation attention 
        self.relation_msg = nn.Parameter(torch.Tensor(self.num_edge_types, n_head, self.k_dim, self.k_dim))
        self.dropout = nn.Dropout(dropout)

        for _ in range(self.num_types):
            self.k_linear.append(nn.Linear(in_dim, out_dim))
            self.q_linear.append(nn.Linear(in_dim, out_dim))
            self.v_linear.append(nn.Linear(in_dim, out_dim))
            self.a_linear.append(nn.Linear(out_dim, out_dim))
        
        # glorot initialization as mention in the paper 
        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, edge_index, x, node_type, edge_type):
        return self.propagate(edge_index, data = x, node_type=node_type, edge_type=edge_type)
    
    def message(self, edge_index, data, node_type, edge_type) -> Tensor:
        # data = graph[n_type].x
        # edge_type = <source_type, edge_type, target_type>

        data_size = data.size(0)
        source_type = self.node_types[node_type]
        target_type = self.node_types[edge_type[2]]

        att = torch.zeros(data_size, self.n_head)
        msg = torch.zeros(data_size, self.n_head, self.k_dim)

        for s_type in range(self.num_types): # itreate over the node types ot get the source
            
            s_mask = (source_type == s_type) # i'm looking at the source type

            if not(s_mask): # skip if looking at the wrong source type
                continue

            k_lin = self.k_linear[s_type] # projection of the attention
            v_lin = self.v_linear[s_type] # projection of the message
            
            for t_type in range(self.num_types): # iterate over the node types to get the target
                t_mask = (target_type == t_type)
                
                if not(t_mask): # skip if looking at the wrong target type
                    continue

                q_lin = self.q_linear[t_type] # porjection of the target 

                for rel_type in range(self.num_edge_types):
                    
                    idx = (self.edge_types[edge_type] == rel_type) & s_mask & t_mask # mask for the edge type
                    if not(idx):
                        continue # skip if worgn relation

                    target = data[t_type]
                    source = data[s_type]
                    
                    # got a problem of dimensionality 
                    Q = q_lin(target).view(-1, self.n_head, self.k_dim)
                    K = k_lin(source).view(-1, self.n_head, self.k_dim)
                    V = v_lin(source).view(-1, self.n_head, self.k_dim)

                    K = torch.bmm(K.transpose(1,0), self.relation_att[rel_type]).transpose(1,0) # computing the key 
                    att[idx] = (Q * K).sum(dim=-1) * self.relation_prior[rel_type]/ self.sqrt_k # attention score
                    msg[idx] = torch.bmm(V.transpose(1,0), self.relation_msg[rel_type]).transpose(1,0) # message
            
            self.att = softmax(att, edge_index)
            out = msg * self.att.view(-1, self.n_head, 1)
            return out.view(-1, self.out_dim)
    
    def update(self, weighted_mgs ,inputs, node_type) -> Tensor:
        '''
        aggregation of the messages and update of the node features
        '''
        messages = F.elu(weighted_mgs) # in the paper they use gelu 
        aggr = torch.zeros(messages.size(0), self.out_dim).to(messages.device)

        for t_type in range(node_type):
            idx = (node_type == t_type)

            if idx.sum() == 0:
                continue

            a_lin = self.a_linear[t_type]
            aggr[idx] = a_lin(messages[idx]) + inputs[idx]
        return aggr

class HGT(nn.Module):
    def __init__(self,graph, hidden_dim, out_dim, n_heads, n_layers, dropout = 0.2):
        super(HGT, self).__init__()

        self.num_nodes = graph.num_nodes
        self.edge_types = graph.edge_types
        self.node_types = graph.node_types
        self.in_dim = 128
        self.node_dict = graph.collect('x') # creating the dictionary of teh nodes
        self.edge_dict = {e_type: graph[e_type] for e_type in graph.edge_types} # creating the dictionary of the edges
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(HGT_Layer(self.n_heads, self.in_dim, self.out_dim, self.edge_types, self.node_types, dropout = dropout))

        for _ in range(n_layers - 1):
            self.layers.append(HGT_Layer(self.n_heads, self.in_dim, self.out_dim, self.edge_types, self.node_types, dropout = dropout))

        self.linear = nn.Linear(hidden_dim, out_dim)
    
    def forward(self, data):
        # x, node_type, edge_index, edge_type = data[key].x, data.node_type, data.edge_index, data.edge_type
        
        for n_type in self.node_types:
            for e_type in self.edge_types:
                for layer in self.layers:
                    x = layer(data[e_type].edge_index, data[n_type].x, n_type, e_type)
                    x = layer.update(x,data[n_type].x, n_type)

        x = self.linear(x)
        return F.log_softmax(x, dim=1)
    

