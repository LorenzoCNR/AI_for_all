
import os
#os.getcwd()
import sys
from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
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
#import tensorflow as tf
#import random

# Imposta seed per numpy
np.random.seed(42)

# Imposta seed per PyTorch
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)  # Per multi-GPU
torch.backends.cudnn.deterministic = True  # Potrebbe ridurre le prestazioni
torch.backends.cudnn.benchmark = False

# Imposta seed per TensorFlow
#tf.random.set_seed(42)

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



def run_hip_models_transform(model_fit, base_path, neural_data):
    #model_fit= joblib.load("cebra_fit.pkl")

    os.chdir(base_path)
    ######################### DA CAMBIARE ##################################
    #### carico dati
    #hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
    
    cebra_posdir3 = model_fit.transform(neural_data)

      
   # return  dd, err_loss, mod1_pred
    return cebra_posdir3
if __name__ == "__main__":
    run_hip_models_transform()

   