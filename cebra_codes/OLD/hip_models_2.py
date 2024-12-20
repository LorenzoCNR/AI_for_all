# -*- coding: utf-8 -*-

import os
os.getcwd()
#import sys
#from pathlib import Path
#import time
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import joblib as jl
import cebra.datasets
from cebra import CEBRA
#from scipy.io import loadmat
#from scipy.io import savemat
#from matplotlib.collections import LineCollection
#from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
#import sklearn.metrics
#import inspect
import torch
#from cebra.datasets.hippocampus import *
#import tensorflow as tf
import random
import math


def run_hip_models(base_path, params):

    # Imposta seed per numpy
    sd = params.get("seed");
    random.seed(sd)
    np.random.seed(int(abs(math.log(random.random()))))
    # Imposta seed per PyTorch
    torch.manual_seed(int(abs(math.log(random.random()))))
    torch.cuda.manual_seed_all(int(abs(math.log(random.random()))))  # Per multi-GPU
    torch.backends.cudnn.deterministic = True  # Potrebbe ridurre le prestazioni
    torch.backends.cudnn.benchmark = False

    # Imposta seed per TensorFlow
    #tf.random.set_seed(42)

    # Imposta seed per il modulo random di Python
    print(sd)


    os.chdir(base_path)
    hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

    
    mod_arch=params.get("model_architecture",'offset10-model')
    out_dim=int(params.get("output_dimension",3))
    temp=int(params.get("temperature",1))
    max_iter=int(params.get("max_iterations", 10000))
    dist=params.get("distance",'cosine')
    cond=params.get("conditional", 'time_delta')
    time_off=int(params.get("time_offsets",10))
    hyb=params.get("hybrid", "").strip('"')
    batch_s=int(params.get("batch_size", 512))
    l_r=float(params.get("learning_rate",3e-4))
    
    
    neural_data=hippocampus_pos.neural
    behavior_data=hippocampus_pos.continuous_index.numpy()
    right=behavior_data[:,1]
    right = right.astype(bool)
    new_behavior_data=behavior_data[:,0]
    print(new_behavior_data)
    new_behavior_data[right]=-new_behavior_data[right];
    print(behavior_data)
    behavior_data=new_behavior_data
    print(behavior_data)
    input('aspetta')   
   # behavior_dic={'dir':behavior_data[:,0],
   #               'right':behavior_data[:,1],
    #              'left':behavior_data[:,2]}
    

    

    cebra_posdir3_model = CEBRA(model_architecture=mod_arch,
                            batch_size=batch_s,
                            learning_rate=l_r,
                            temperature=temp, 
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off,
                            hybrid=hyb)
    
    cebra_posdir3_model.fit(neural_data,behavior_data)
    cebra_posdir3 = cebra_posdir3_model.transform(neural_data) 
    
    return cebra_posdir3, behavior_data
if __name__ == "__main__":
    run_hip_models()




#### plot in matlab

