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



def run_hip_models(base_path):

    os.chdir(base_path)
    ######################### DA CAMBIARE ##################################
    #### carico dati
    hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

    
    
    ###### Load model hyperparameters
    try:
    
        data_param=loadmat('params.mat')
        
        params = data_param['params']
        
        mod_arch= params['mod_arch'][0].item() 
        mod_arch= mod_arch[0]
        
        out_dim= int(params['output_dimension'][0][0])
        
        temp=int(params['temperature'][0][0])
        
        max_iter=int(params['max_iter'][0][0])
        
        dist= params['distance'][0].item() 
        dist= dist[0]
        
        cond= params['conditional'][0].item() 
        cond= cond[0]
        
        time_off=int(params['time_offsets'][0][0])
    
    except:
        mod_arch='offset10-model'
        out_dim=3
        temp=1
        max_iter=10000
        dist='cosine'
        cond='time_delta'
        time_off=10
    
    
   
    
    #max_iterations = 10 ## defaut 5000
    
    neural_data=hippocampus_pos.neural
    behavior_data=hippocampus_pos.continuous_index.numpy()
    
    
    behavior_dic={'dir':behavior_data[:,0],
                  'right':behavior_data[:,1],
                  'left':behavior_data[:,2]}
    

    

    cebra_posdir3_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp, 
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
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

