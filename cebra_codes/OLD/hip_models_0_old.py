# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#### Mettere la directory di interesse la stessa di matlab


#!pip install ripser
#import ripser
#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')
#base_dir=r'/home/zlollo/CNR/Cebra_for_all'
#os.chdir(base_dir)

import os
os.getcwd()
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
from scipy.io import loadmat
from scipy.io import savemat
#from dataset import SingleRatDataset  # Assumendo che il codice sia in 'dataset.py'
from matplotlib.collections import LineCollection
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import inspect
import torch
from cebra.datasets.hippocampus import *
import tensorflow as tf
import random

# Imposta seed per numpy
np.random.seed(42)

# Imposta seed per PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # Per multi-GPU
torch.backends.cudnn.deterministic = True  # Potrebbe ridurre le prestazioni
torch.backends.cudnn.benchmark = False

# Imposta seed per TensorFlow
tf.random.set_seed(42)

# Imposta seed per il modulo random di Python
random.seed(42)


 #from tensorflow.python.client import device_lib
    
'''
 # Verifica la disponibilità di GPU
    if tf.test.is_gpu_available():
        print("TensorFlow sta utilizzando una GPU.")
    else:
        print("TensorFlow non sta utilizzando una GPU.")
    
    
    print(device_lib.list_local_devices())
    
    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]
    with tf.device('/device:GPU:0'):
        print(get_available_devices())
  '''



def run_hip_models(base_path, params):

    os.chdir(base_path)
    ######################### DA CAMBIARE ##################################
    #### carico dati
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
    
    # #d_vis_hyp=data["visualization"]['hypothesis']
    
    
    # hypoth={"embedding": cebra_posdir3, "label": behavior_data}
    

    # cebra_time3_model = CEBRA(model_architecture=mod_arch,
    #                         batch_size=512,
    #                         learning_rate=3e-4,
    #                         temperature=temp,
    #                         output_dimension=out_dim,
    #                         max_iterations=max_iter,
    #                         distance=dist,
    #                         conditional='time',
    #                         device='cuda_if_available',
    #                         verbose=True,
    #                         time_offsets=time_off)
    
    # cebra_time3_model.fit(neural_data)
    # cebra_time3 = cebra_time3_model.transform(neural_data)
    
    # ttime={"embedding": cebra_time3, "label": behavior_data}
    
    
    # ### modello ibrido con info temporali e posizionali 
    # cebra_hybrid_model = CEBRA(model_architecture=mod_arch,
    #                         batch_size=512,
    #                         learning_rate=3e-4,
    #                         temperature=temp,
    #                         output_dimension=out_dim,
    #                         max_iterations=max_iter,
    #                         distance=dist,
    #                         conditional=cond,
    #                         device='cuda_if_available',
    #                         verbose=True,
    #                         time_offsets=time_off,
    #                         hybrid = True)
    
    # cebra_hybrid_model.fit(neural_data, behavior_data)
    # cebra_hybrid = cebra_hybrid_model.transform(neural_data)
    
    # hhybrid={"embedding": cebra_hybrid, "label": behavior_data}
    
    
    ############################## Questo in matlab ###############################
    
    
    
    ##
    
    
   # return  dd, err_loss, mod1_pred
    return cebra_posdir3, behavior_data
if __name__ == "__main__":
    run_hip_models()




#### plot in matlab

