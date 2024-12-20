# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#### Mettere la directory di interesse la stessa di matlab
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

def run_hip_models_fit(base_path, params, neural_data, labels, output_folder):
    os.chdir(base_path)

    # load parameters
    mod_arch = params.get("model_architecture", 'offset10-model')
    out_dim = int(params.get("output_dimension", 3))
    temp = int(params.get("temperature", 1))
    max_iter = int(params.get("max_iterations", 10000))
    dist = params.get("distance", 'cosine')
    cond = params.get("conditional", 'time_delta')
    time_off = int(params.get("time_offsets", 10))
    hyb = params.get("hybrid",False)
    batch_s = int(params.get("batch_size", 512))
    l_r = float(params.get("learning_rate", 3e-4))

    # 
    cebra_model = CEBRA(model_architecture=mod_arch, batch_size=batch_s,
                        learning_rate=l_r, temperature=temp, output_dimension=out_dim,
                        max_iterations=max_iter, distance=dist, conditional=cond,
                        device='cuda_if_available', verbose=True, time_offsets=time_off, hybrid=hyb)

    # fit the model
    cebra_model.fit(neural_data, labels)

    # Save model
    model_path = os.path.join(output_folder, "cebra_fit.pkl")
    jl.dump(cebra_model, model_path)
    
    return model_path
if __name__ == "__main__":
    run_hip_models_fit()





