# DRLFS-master
Created by Hang Zhang from Northwestern Polytechnical University. 

## Introduction
DRLFS is a novel deep reinforcement learning (DRL) framework for intersecting machining feature segmentation. The framework takes intersecting machining features represented by attributed adjacency graphs (AAGs) as input and isolated machining features as output. 

For confidentiality reasons, the original codes and data used in our experiments cannot be made public. However, we provide modified versions of the codes and data used. While it may not be possible to replicate the experiment exactly, it is possible to use modified codes and data to accomplish the main tasks of the experiment and obtain similar results. 

## Setup
(1)	cuda 11.6.112
(2)	python 3.8.13  
(3)	pytorch 1.12.0  
(4)	tianshou 0.4.11
(5) dgl 0.9.1
(6) gym 0.25.2
(7) tensorboard 2.10.0

The code is tested on Intel Core i9-10980XE CPU, 128GB memory, and NVIDIA GeForce RTX 3090 GPU. 

## Train
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.  
(2)	Create the folder named `logdir` in the root directory.  
(3)	Download related point cloud [datasets](https://drive.google.com/drive/folders/1ux1-LsM1O7J3ufHFS5a0BlARX1qIEP1d?usp=sharing) (traindata.h5, validationdata.h5).   
(4)	Put the datasets in the folder `data`.  
(5)	Run `python train.py` to train the network.  

## Test 
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.  
(2)	Download related  point cloud [datasets](https://drive.google.com/drive/folders/1ux1-LsM1O7J3ufHFS5a0BlARX1qIEP1d?usp=sharing) (testdata.h5).   
(3)	Put the datasets in the folder `data`.  
(4)	Download related pre-trained ASIN [model](https://drive.google.com/drive/folders/1Ha-Q2G3AzqQI4RZEB_18ZAZIYyMx1FPb?usp=sharing) (.h5). **The pre-trained ASIN model was trained on the publicly available dataset.**  
(5)	Put pre-trained ASIN model in the folder `models`.  
(6)	Run `python test.py` to test the network.  

## Predict
(1)	Get the ASIN source code by cloning the repository: https://github.com/HARRIXJANG/ASIN-master.git.  
(2)	Download related pre-trained ASIN [model](https://drive.google.com/drive/folders/1Ha-Q2G3AzqQI4RZEB_18ZAZIYyMx1FPb?usp=sharing) (.h5). **The pre-trained ASIN model was trained on the publicly available dataset.**  
(3)	Put the pre-trained ASIN model in the folder `models`.  
(4)	Run `python predict.py` to predict a part. The result is a text file (.txt). In this text file, the first line is "start", the second line is the name of the part, the third line is the tag number of each face in the part (generated by catia), the fourth line is the category (hole, pocket) corresponding to each face, the fifth line is the identification of the bottom surface (0 represents a non-bottom face, 1 represents a bottom face), the sixth line is the clustering results (each set of faces represents a machining feature), and the eighth line is "end".  

## Citation
If you use this code please cite:  
```
@inproceedings{zhang2022asin,  
      title={Machining feature recognition based on a novel multi-task deep learning network},  
      author={Hang Zhang, Shusheng Zhang, Yajun Zhang, Jiachen Liang, and Zhen Wang},  
      booktitle={Robotics and Computer-Integrated Manufacturing},  
      year={2022}  
    }
``` 
If you have any questions about the code, please feel free to contact me (2015301048@mail.nwpu.edu.cn).