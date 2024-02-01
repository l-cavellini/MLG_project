import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
import math


class HGT_Layer(MessagePassing):
    def __init__(self, n_head, in_dim, out_dim, num_relations, num_types, dropout = 0.2, **kwargs):
        super(HGT_Layer, self).__init__(node_dim=0, aggr='add', **kwargs)

        self.n_head = n_head
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.k_dim = out_dim // n_head
        self.sqrt_k = math.sqrt(self.k_dim)
        self.att = None

        self.k_linear = nn.ModuleList()
        self.q_linear = nn.ModuleList()
        self.v_linear = nn.ModuleList()
        self.a_linear = nn.ModuleList()

        self.relation_prior = nn.Parameter(torch.ones(num_relations, n_head)) # for each relation
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_head, self.k_dim, self.k_dim)) # for each relation, 
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_head, self.k_dim, self.k_dim))
        self.dropout = nn.Dropout(dropout)

        for t in range(num_types):
            self.k_linear.append(nn.Linear(in_dim, out_dim))
            self.q_linear.append(nn.Linear(in_dim, out_dim))
            self.v_linear.append(nn.Linear(in_dim, out_dim))
            self.a_linear.append(nn.Linear(out_dim, out_dim))
        
        # glorot initialization as mention in the paper 
        glorot(self.relation_att)
        glorot(self.relation_msg)

    def forward(self, x, node_type, edge_index, edge_type):
        return self.propagate(edge_index, x=x, node_type=node_type, edge_type=edge_type)
    
    def message(self, edge_index_i, node_i, node_j, node_type_i, node_type_j, edge_type) -> Tensor:
        '''
        node_j -> node_i
        '''
        data_size = edge_index_i.size(0)
        
        att = torch.zeros(data_size, self.n_head).to(node_i.device)
        msg = torch.zeros(data_size, self.n_head, self.k_dim).to(node_i.device)

        for s_type in range(self.num_types): # itreate over the node types ot get the source
            
            s_mask = (node_type_i == s_type)
            k_lin = self.k_linear[s_type] # projection of the attention
            v_lin = self.v_linear[s_type] # projection of the message
            
            for t_type in range(self.num_types): # iterate over the node types to get the target
                t_mask = (node_type_j == t_type)
                q_lin = self.q_linear[t_type] # porjection of the target 

                for rel_type in range(self.num_relations):
                    
                    idx = (edge_type == int(rel_type)) & s_mask & t_mask # mask for the edge type
                    if idx.sum() == 0:
                        continue # skip if there's no such relation

                    target = node_i[idx]
                    source = node_j[idx]

                    Q = q_lin(target).view(-1, self.n_head, self.k_dim)
                    K = k_lin(source).view(-1, self.n_head, self.k_dim)
                    V = v_lin(source).view(-1, self.n_head, self.k_dim)

                    K = torch.bmm(K.transpose(1,0), self.relation_att[rel_type]).transpose(1,0) # computing the key 
                    att[idx] = (Q * K).sum(dim=-1) * self.relation_prior[rel_type]/ self.sqrt_k # attention score
                    msg[idx] = torch.bmm(V.transpose(1,0), self.relation_msg[rel_type]).transpose(1,0) # message
            
            self.att = softmax(att, edge_index_i)
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

         