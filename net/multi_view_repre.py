# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 15:06:20 2021

@author: danli
"""
import torch
import torch.nn as nn
from net.utils import relaition_matrix,preprocess_graph
from net.multi_view_network import S,E,GCNModelAE

class Multi_view_dynamic_relation(nn.Module):
    def __init__(self):
        super(Multi_view_dynamic_relation, self).__init__()
        #50 50

        self.S_net1 = S(3000,3000,3000);
        self.E_net1 = E(3000,500,200);
        ##########################
        self.S_net2 = S(3000,3000,3000);
        self.E_net2 = E(3000,500,200);
        #########################
        self.S_net3 = S(3000,3000,3000);
        self.E_net3 = E(3000,500,200);
        #########################
        self.S_net4 = S(3000,3000,3000);
        self.E_net4 = E(3000,500,200);
        ####################
        self.S_net5 = S(3000,3000,3000);
        self.E_net5 = E(3000,500,200);
        ############################
        self.relation_net=GCNModelAE(200,100,100,0)
        
        

    def forward(self,x1,x2,x3,x4,x5,view_num):
        mask_s1_pro= self.S_net1(x1);
        x1_s=x1* mask_s1_pro
        x1_en=self.E_net1(x1_s)
        x1_relation = relaition_matrix(x1)
        x1_en_relation = relaition_matrix(x1_en)
        ######################################
        mask_s2_pro= self.S_net2(x2);
        x2_s=x2* mask_s2_pro
        x2_en=self.E_net2(x2_s)
        x2_relation = relaition_matrix(x2)
        x2_en_relation = relaition_matrix(x2_en)
        #########################################
        mask_s3_pro= self.S_net3(x3)
        x3_s=x3* mask_s3_pro
        x3_en=self.E_net3(x3_s) 
        x3_relation = relaition_matrix(x3)
        x3_en_relation = relaition_matrix(x3_en)
        ##########################################
        mask_s4_pro= self.S_net4(x4);
        x4_s=x4* mask_s4_pro
        x4_en=self.E_net4(x4_s)
        x4_relation = relaition_matrix(x4)
        x4_en_relation = relaition_matrix(x4_en)
        ############################################
        mask_s5_pro= self.S_net5(x5);
        x5_s=x5* mask_s5_pro
        x5_en=self.E_net5(x5_s)
        x5_relation = relaition_matrix(x5)
        x5_en_relation = relaition_matrix(x5_en)
        ##########################################
        MMfeature = torch.cat([x1_en,x2_en,x3_en,x4_en,x5_en], dim=0) #图卷积网络的样本
        num_=x1_en.size(0)
        adj_view=torch.zeros((view_num*num_,view_num*num_)).cuda()
        for view in range(1,view_num+1):
            adj_view[(view-1)*num_:view*num_,(view-1)*num_:view*num_]=locals()['x{0}_en_relation'.format(view)]
        for ii in range(1,view_num):
            for jj in range(ii+1,view_num+1):
                 view_inter=((locals()['x{0}_en_relation'.format(ii)]>=0.5).float())*((locals()['x{0}_en_relation'.format(jj)]>=0.5).float())
                 adj_view[(ii-1)*num_:ii*num_,(jj-1)*num_:jj*num_]=locals()['x{0}_en_relation'.format(jj)]*view_inter
                 adj_view[(jj-1)*num_:jj*num_,(ii-1)*num_:ii*num_]=locals()['x{0}_en_relation'.format(ii)]*view_inter
        adj_label=adj_view
        adj_norm=preprocess_graph(adj_view)          
        x_view_relation,adj_recover=self.relation_net(MMfeature,adj_norm)
       
                      

        #########################################
        
        return mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
               x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
               x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
               x_view_relation,adj_label,adj_recover
              