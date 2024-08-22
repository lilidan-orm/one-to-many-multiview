# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:16:34 2021

@author: danli
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys
import time
import itertools
from multi_view_data import *
import numpy as np
from sklearn.cluster import KMeans
from net.metrics import get_avg_acc,get_avg_nmi,get_avg_RI,get_avg_f1
from torch.utils.data import DataLoader
import scipy.io as sio
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0'
use_cuda = torch.cuda.is_available()
sys.path.append('.');
import warnings
warnings.filterwarnings("ignore")
from loss.multi_view_loss import MultiView_all_loss;
from net.multi_view_repre import Multi_view_dynamic_relation;

batch_size =256
epoch_num1=20
epoch_num2 =50
test_batch_size =256


###################################

#########################################################################

def train():
    train_dataset_paired=Multimodal_Datasets(train=True)
    test_dataset=Multimodal_Datasets(train=False)

    trainloader_paired = DataLoader(train_dataset_paired, batch_size,
                                         shuffle=True, num_workers=0)
    testloader=DataLoader(test_dataset, test_batch_size,
                                         shuffle=False, num_workers=0)# clustering training_set=test_set
    print('Finish loading the data....')
    net =Multi_view_dynamic_relation();
    lossfunc1 = MultiView_all_loss(round=1);
    optimizer_1 = torch.optim.Adam(itertools.chain(net.S_net1.parameters(),net.E_net1.parameters(),\
                                                net.S_net2.parameters(),net.E_net2.parameters(),\
                                                net.S_net3.parameters(),net.E_net3.parameters(),\
                                                net.S_net4.parameters(),net.E_net4.parameters(),\
                                                net.S_net5.parameters(),net.E_net5.parameters()),lr=0.0001) 
    
    print('-' * 10,'Pre-Training Start','-' * 10)
    for epoch in range(epoch_num1):
        num_batches = len(trainloader_paired)
        running_loss_1,proc_size = 0,0
        for batch_idx,train_data in enumerate(trainloader_paired):
            inputs_x1, inputs_x2,inputs_x3,inputs_x4,inputs_x5,label_train= train_data
                
            optimizer_1.zero_grad()
            if use_cuda:
                inputs_x1,inputs_x2, inputs_x3,inputs_x4,inputs_x5, label_train= inputs_x1.cuda(),inputs_x2.cuda(),inputs_x3.cuda(),inputs_x4.cuda(),inputs_x5.cuda(),label_train.cuda(non_blocking=True)
                net = net.cuda()
                mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
                x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
                x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
                _,_,_=net(inputs_x1,inputs_x2,inputs_x3,inputs_x4,inputs_x5,5)
            
            loss1= lossfunc1(mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
                x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
                x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
                _,_,_)
            
            loss_1=loss1
            loss_1.backward()
            optimizer_1.step()
            running_loss_1 += loss_1.item()
        
            
            proc_size += batch_idx
            if batch_idx % 100 == 0:  
                if batch_idx==0:
                    avg_loss_1= running_loss_1      
                else:
                    avg_loss_1 =running_loss_1 / 100
                
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Train Loss_1 {:5.4f}'.
                        format(epoch,batch_idx, num_batches, avg_loss_1))
                running_loss_1, proc_size = 0, 0
                
    print('-' * 10,'Second Training Start','-' * 10)
    lossfunc = MultiView_all_loss();
    optimizer_2 = torch.optim.Adam(net.relation_net.parameters(),lr=0.0001) 
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=3, gamma=0.2)  
    ##########################################################################################
    for epoch in range(epoch_num2):
        net.train()
        #scheduler.step()
        num_batches = len(trainloader_paired)
        running_loss_1,running_loss_2, proc_size = 0,0,0
        start_time = time.time()
        for batch_idx,train_data in enumerate(trainloader_paired):
            inputs_x1, inputs_x2,inputs_x3,inputs_x4,inputs_x5, label_train= train_data
                
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            if use_cuda:
                inputs_x1,inputs_x2, inputs_x3,inputs_x4,inputs_x5,label_train= inputs_x1.cuda(),inputs_x2.cuda(),inputs_x3.cuda(),inputs_x4.cuda(),inputs_x5.cuda(),label_train.cuda(non_blocking=True)
                net = net.cuda()
                mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
                x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
                x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
                x_view_relation,adj_label,adj_recover=net(inputs_x1,inputs_x2,inputs_x3,inputs_x4,inputs_x5,5)
            
            loss, loss1,loss2 = lossfunc(mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
                x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
                x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
                x_view_relation,adj_label,adj_recover)
            
        
        
            loss_1=loss1
            loss_1.backward(retain_graph=True)
            optimizer_1.step()
            running_loss_1 += loss_1.item()

            loss_2=loss2
            loss_2.backward()
            optimizer_2.step()
            running_loss_2 += loss_2.item()

        
            
            proc_size += batch_idx
            if batch_idx % 100 == 0:  
                if batch_idx==0:
                    avg_loss_1= running_loss_1
                    avg_loss_2= running_loss_2 
                else:
                    avg_loss_1 =running_loss_1 / 100
                    avg_loss_2 =running_loss_2 / 100
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss_1 {:5.4f}| Train Loss_2 {:5.4f}'.
                        format(epoch,batch_idx, num_batches, elapsed_time * 1000 / 100, avg_loss_1,avg_loss_2))
                running_loss_1,running_loss_2, proc_size = 0, 0, 0
                start_time = time.time()
        
    print('-' * 10,'Test Start','-' * 10)
    x1_feature = []
    x2_feature = []
    x3_feature=[]
    x4_feature=[]
    x5_feature=[]
    x_view = []
    label_all=[]
    mask_s1_all=[]
    mask_s2_all=[]
    mask_s3_all=[]
    mask_s4_all=[]
    mask_s5_all=[]
    net.eval()
    with torch.no_grad():
        for data in testloader:
            x_view = []
            x1,x2,x3,x4,x5,labels= data
            mask_s1_pro,mask_s2_pro,mask_s3_pro,mask_s4_pro,mask_s5_pro,\
            x1_relation,x2_relation,x3_relation,x4_relation,x5_relation,\
            x1_en_relation,x2_en_relation,x3_en_relation,x4_en_relation,x5_en_relation,\
            x_view_relation,adj_label,adj_recover= net(x1.cuda(),x2.cuda(),x3.cuda(),x4.cuda(),x5.cuda(),5)
            for view in range(5):
                x_view.append(x_view_relation[view*x1.size(0):(view+1)*x1.size(0),:])
            x1_feature.append(x_view[0])
            x2_feature.append(x_view[1])    
            x3_feature.append(x_view[2])  
            x4_feature.append(x_view[3])  
            x5_feature.append(x_view[4]) 
            label_all.append(labels)
            ############################
            # mask_s1_all.append(mask_s1_pro)
            # mask_s2_all.append(mask_s2_pro)
            # mask_s3_all.append(mask_s3_pro)
            # mask_s4_all.append(mask_s4_pro)
            # mask_s5_all.append(mask_s5_pro)
            
                    
    x_view_final =(torch.cat(x1_feature).cpu().numpy()+torch.cat(x2_feature).cpu().numpy()+torch.cat(x3_feature).cpu().numpy()+torch.cat(x4_feature).cpu().numpy()+torch.cat(x5_feature).cpu().numpy())/5

    label_all = torch.cat(label_all).cpu().numpy()
    label_all=label_all.reshape(len(label_all),)


    estimator=KMeans(6)
    label_pred= estimator.fit_predict(x_view_final)
    return label_pred,label_all
times=30
pred_all=[]
for t in range(times):
    print('-' * 10,t+1,'-' * 10)
    label_pred,label_all=train()
    pred_all.append(label_pred)
    
acc_avg, acc_std = get_avg_acc(label_all, pred_all, times)
nmi_avg, nmi_std = get_avg_nmi(label_all, pred_all, times)
ri_avg, ri_std = get_avg_RI(label_all, pred_all, times)
f1_avg, f1_std = get_avg_f1(label_all, pred_all, times)

print('acc: {acc:.4f}\t'
                'acc_std: {acc_std:.4f}\t'
                'NMI: {NMI:.4f}\t'
                'NMI_std: {nmi_std:.4f}\t'
                'F:{F:.4f}\t'
                'F_std: {f1_std:.4f}\t'
                'RI:{RI:.4f}\t'
                'RI_std: {ri_std:.4f}'.format(acc=acc_avg,acc_std=acc_std,NMI=nmi_avg,nmi_std=nmi_std,F=f1_avg,f1_std=f1_std,RI=ri_avg,ri_std=ri_std))

# print('Finished Training') 
#####################################################
