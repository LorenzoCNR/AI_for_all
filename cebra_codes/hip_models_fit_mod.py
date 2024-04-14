# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import sys
import h5py

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
from matplotlib.collections import LineCollection
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import inspect
import torch
if len(sys.argv) < 2:
    print("Too few args!!!")

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

def set_seeds(seed):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




def run_hip_models_fit(hdf5_path):
 
    with h5py.File(hdf5_path, 'r') as hdf:
        # Load data from datasets
        neural_data = hdf['/Achilles_data/neural'][:]
        labels = hdf['/Achilles_data/behavior'][:]
        # Read model parameters and other settings from attributes
        model_params = {
            'model_architecture': hdf.attrs['model_architecture'],
            'batch_size': int(hdf.attrs['batch_size']),
            'learning_rate': float(hdf.attrs['learning_rate']),
            'temperature': int(hdf.attrs['temperature']),
            'output_dimension': int(hdf.attrs['output_dimension']),
            'max_iterations': int(hdf.attrs['max_iterations']),
            'distance': hdf.attrs['distance'],
            'conditional': hdf.attrs['conditional'],
            'time_offsets': int(hdf.attrs['time_offsets']),
            'hybrid': hdf.attrs['hybrid'],
            'verbose': hdf.attrs['verbose']
        }


        model_output_path = hdf.attrs['model_output_path']
        seed = hdf.attrs['seed']
        set_seeds(seed)

        # Initialize and configure the CEBRA model
        cebra_model = CEBRA(**model_params)
        
        # Fit the model according to the specified type
        model_type = hdf.attrs.get('model_type', 'hypothesis')
        if model_type == 'hypothesis':
            cebra_model.fit(neural_data, labels)
        elif model_type == 'discovery':
            cebra_model.fit(neural_data)
        elif model_type == 'shuffle':
            shuffled_labels = np.random.permutation(labels)
            cebra_model.fit(neural_data, shuffled_labels)

        jl.dump(cebra_model, model_output_path)
        print(f"Model saved at {model_output_path}")

if __name__ == "__main__":
    hdf5_path = sys.argv[1]
    run_hip_models_fit(hdf5_path)