# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:43:56 2021

@author: danli
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd.function import Function



import sys
sys.path.append('.');

class MultiView_all_loss(nn.Module):
    def __init__(self,round=2):

        super(MultiView_all_loss, self).__init__()
        self.loss_fn = nn.MSELoss(reduce=True, size_average=True)
        self.round=round
        
    def forward(self, mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover):
       
        mask_loss=torch.mean(mask_s1_pro)+torch.mean(mask_s2_pro)+torch.mean(mask_s3_pro)+torch.mean(mask_s4_pro)+\
                  torch.mean(mask_s5_pro)
        N=x1_relation.size(0)
        mask_dui_jiao_0 = (torch.ones(N ,N) - torch.eye(N, N)).cuda()
        dynamic_relation_loss= torch.FloatTensor([0]).cuda();
        for view in range(1,5+1):
            locals()['x{0}_relation'.format(view)]=locals()['x{0}_relation'.format(view)]*mask_dui_jiao_0 
            locals()['x{0}_en_relation'.format(view)]=locals()['x{0}_en_relation'.format(view)]*mask_dui_jiao_0 
            dynamic_relation_loss=dynamic_relation_loss+\
                                  torch.mean(torch.pow(torch.clamp(locals()['x{0}_relation'.format(view)]*((locals()['x{0}_relation'.format(view)]>=0.5).float())-locals()['x{0}_en_relation'.format(view)]*((locals()['x{0}_en_relation'.format(view)]>=0.5).float()), min=0.0), 2)) +\
                                  torch.mean(torch.pow(locals()['x{0}_relation'.format(view)]*((locals()['x{0}_relation'.format(view)]<0.5).float())*((locals()['x{0}_relation'.format(view)]>=0.2).float())-locals()['x{0}_en_relation'.format(view)]*((locals()['x{0}_en_relation'.format(view)]<0.5).float())*((locals()['x{0}_en_relation'.format(view)]>=0.2).float()), 2)) +\
                                  torch.mean(torch.pow(torch.clamp(locals()['x{0}_en_relation'.format(view)]*((locals()['x{0}_en_relation'.format(view)]<0.2).float())-locals()['x{0}_relation'.format(view)]*((locals()['x{0}_relation'.format(view)]<0.2).float()), min=0.0),2))  
        loss1=mask_loss+dynamic_relation_loss
        ########################################deocder_loss
        loss2 = F.binary_cross_entropy_with_logits(adj_recover, adj_label)
        
        if self.round==1:
            return loss1
        else:
            loss=loss1+loss2
            return loss,loss1,loss2

        # for view in range(1,view_num+1):
        #     locals()['x{0}_view_relation'.format(ii)]= x_view_relation[(ii-1)*num_:,ii*num_]
        # x_view_final=x1_view_relation+x2_view_relation+x3_view_relation+x4_view_relation