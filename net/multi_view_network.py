# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:42:56 2021

@author: danli
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
#from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

#selector-mask
class S(nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(S, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),nn.BatchNorm1d(n_hidden_1), nn.Sigmoid())
        #self.layer3 = nn.Sequential(nn.Linear(n_hidden_1, out_dim), nn.Sigmoid())
        #self.dropout=nn.Dropout(0.1)
    def forward(self, x):
        x1 = self.layer1(x)
        #mask_s = self.layer3(x1)
        return x1

#encoder-view-specific
class E(nn.Module):
    def __init__(self, in_dim, n_hidden_1,out_dim):
        super(E, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1),nn.Tanh())
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1,out_dim), nn.BatchNorm1d(out_dim))
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x2

#GCN-layer
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.sparse.mm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

#GCN-network

class GCNModelAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def forward(self, x, adj):
        x1 = self.gc1(x, adj)
        x2 = self.gc2(x1, adj)

        return x2, self.dc(x2)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj






