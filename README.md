One-to-Many Multi-view
=
This repository contains the code for our paper "Multi-view Representation Learning with One-to-Many Dynamic Relationships".

Requirment
=
pytoch>=1.3.1  
python>=3.7.4  
numpy>=1.16.5  
scikit-learn>=0.21.3  
scipy>=1.3.1  

Dataset
=
The datasets used in the paper can be obtained through the following links [Google Drive](https://drive.google.com/file/d/1gKAcKx3Gu2zzJieXfg7r1Dm4dveqGH84/view?usp=sharing) or  [Baidu Disk](https://pan.baidu.com/s/1a1BVH26zIcer_Qa8f-BbrA?pwd=g84z) (code: g84z)

Usage
=
The code includes:  
* an example implementation of the model on the Reuters dataset
* an example clustering task  

You can get the following output by running multi_view_main.py  
(modify the dataset path in the multi_view_data.py)




```
Epoch 43 | Batch   0/ 74 | Time/Batch(ms)  2.11 | Train Loss_1 6.4164| Train Loss_2 0.6931
Epoch 44 | Batch   0/ 74 | Time/Batch(ms)  2.15 | Train Loss_1 6.3914| Train Loss_2 0.6934
Epoch 45 | Batch   0/ 74 | Time/Batch(ms)  2.18 | Train Loss_1 6.4116| Train Loss_2 0.6933
Epoch 46 | Batch   0/ 74 | Time/Batch(ms)  2.09 | Train Loss_1 6.3757| Train Loss_2 0.6932
Epoch 47 | Batch   0/ 74 | Time/Batch(ms)  2.19 | Train Loss_1 6.3719| Train Loss_2 0.6936
Epoch 48 | Batch   0/ 74 | Time/Batch(ms)  2.12 | Train Loss_1 6.3610| Train Loss_2 0.6932
Epoch 49 | Batch   0/ 74 | Time/Batch(ms)  2.11 | Train Loss_1 6.3502| Train Loss_2 0.6938
---------- Test Start ----------
acc: 0.8879     acc_std: 0.0859 NMI: 0.8623     NMI_std: 0.0514 F:0.8817        F_std: 0.0560   RI:0.9477       RI_std: 0.0293
```
Reference
=
If you find our work useful in your research, please consider citing:
```
@article{li2024multiview,
  title={Multiview Representation Learning With One-to-Many Dynamic Relationships},
  author={Li, Dan and Wang, Haibao and Ying, Shihui},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
