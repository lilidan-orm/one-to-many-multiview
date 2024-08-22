# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:56:12 2021

@author: danli
"""

import numpy as np
import torch
import scipy.sparse as sp

def relaition_matrix (x):
  N=x.size(0)
  tmp = torch.norm(x, 2, 1);
  x_norm = x / tmp.view(N, -1);
  x_relation=torch.mm( x_norm,x_norm.t())  
   #x_relation=x_relation*(x_relation>=0).float()
  x_relation=torch.clamp(x_relation,min=0.0)
  return x_relation

def preprocess_graph(adj):
    D = torch.pow(adj.sum(1).float(), -1).flatten()
    D = torch.diag(D)
    adj = torch.matmul(D,adj)
    return torch_sparse_tensor(adj)


def torch_sparse_tensor(adj_all_view):
    
    idx = torch.nonzero(adj_all_view).T  
    data = adj_all_view[idx[0],idx[1]]
    #coo_a = torch.sparse_coo_tensor(idx, data, adj_all_view.shape)
    return torch.sparse.FloatTensor(idx, data, adj_all_view.shape)