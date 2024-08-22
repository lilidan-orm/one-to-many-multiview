import random
import sys
import scipy.io as sio
import numpy as np
import torch
from PIL import Image
import os
import torch.utils.data as data
from torchvision import transforms
from sklearn import preprocessing
import h5py
#from keras.utils import np_utils
from sklearn.decomposition import PCA
# 


class Multimodal_Datasets(data.Dataset):
    def __init__(self, train=True):
        self.train = train
        data_multiview1 = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters_pca/data_view1.mat")
        data_view1=data_multiview1['X1']
       
        data_multiview2 = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters_pca/data_view2.mat")
        data_view2=data_multiview2['X2']
        
        data_multiview3 = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters_pca/data_view3.mat")
        data_view3=data_multiview3['X3']
        
        data_multiview4 = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters_pca/data_view4.mat")
        data_view4=data_multiview4['X4']
        
        data_multiview5 = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters_pca/data_view5.mat")
        data_view5=data_multiview5['X5']
        data_multiview = sio.loadmat("C:/Users/B/Desktop/paper_code/multi-view_representation_learning/dataset_clustering/Reuters.mat")
        label=data_multiview['Y']
        min_max_scaler_1 = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data_view1)
        data_view1=min_max_scaler_1.transform(data_view1)
        min_max_scaler_2 = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data_view2)
        data_view2=min_max_scaler_2.transform(data_view2)
        min_max_scaler_3 = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data_view3)
        data_view3=min_max_scaler_3.transform(data_view3)
        min_max_scaler_4 =preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data_view4)
        data_view4=min_max_scaler_4.transform(data_view4)
        min_max_scaler_5 = preprocessing.MinMaxScaler(feature_range=(0,1)).fit(data_view5)
        data_view5=min_max_scaler_5.transform(data_view5)
         

        
        if self.train:
            self.train_data_view1 = data_view1
            self.train_data_view2 = data_view2
            self.train_data_view3 = data_view3
            self.train_data_view4 = data_view4
            self.train_data_view5 = data_view5
            self.train_labels = label
           
           
        else:
            self.test_data_view1= data_view1
            self.test_data_view2= data_view2
            self.test_data_view3= data_view3
            self.test_data_view4= data_view4
            self.test_data_view5= data_view5
            self.test_labels = label
           
    def __getitem__(self, index):
        if self.train:
            view1= self.train_data_view1[index]
            view2=self.train_data_view2[index]
            view3=self.train_data_view3[index]
            view4=self.train_data_view4[index]
            view5=self.train_data_view5[index]
            target = self.train_labels[index]
            
        else:
            view1= self.test_data_view1[index]
            view2=self.test_data_view2[index]
            view3=self.test_data_view3[index]
            view4=self.test_data_view4[index]
            view5=self.test_data_view5[index]
            target = self.test_labels[index]
            
       
        view1 = torch.from_numpy(view1).float()
        view2 = torch.from_numpy(view2).float()
        view3 = torch.from_numpy(view3).float()
        view4 = torch.from_numpy(view4).float()
        view5 = torch.from_numpy(view5).float()
       
        
        
        target = target.astype(np.int64)
        return view1, view2,view3,view4,view5,target
    
    def __len__(self):
        if self.train:
            return len(self.train_labels)
        else:
            return len(self.test_labels)
