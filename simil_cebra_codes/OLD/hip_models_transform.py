
import os
#os.getcwd()
import sys
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
import joblib as jl
import h5py
from datetime import datetime
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
#import tensorflow as tf
#import random

# numpy seed
np.random.seed(42)

# Pytorch seed
torch.manual_seed(42)
# multi-GPU seed
torch.cuda.manual_seed_all(42)  
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

# TF seed
#tf.random.set_seed(42)

# Seed random module python
random.seed(42)


 #from tensorflow.python.client import device_lib
    
'''
 # Verify gpus
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

'''
def update_hdf5_attributes(hdf5_path, updates):
    with h5py.File(hdf5_path, 'r+') as hdf:  
        for key, value in updates.items():
            hdf.attrs[key] = value  
        print("Updated attributes:")
        for key in updates:
            print(f"{key}: {hdf.attrs[key]}") 


hdf5_path = 'our_hdf'
updates = {
    # modify an attribute
    'batch_size': 256,  
     # Add attribute
    'new_attribute': 'value' 
}
update_hdf5_attributes(hdf5_path, updates)
'''



def run_hip_models_transform(model_path, neural_data):
    # load model
    model_fit = jl.load(model_path)
    #seed=42
    #seed = set_seeds(seed)

        # 
    transformed_data = model_fit.transform(neural_data)

    return transformed_data

if __name__ == "__main__":
    run_hip_models_transform()



   