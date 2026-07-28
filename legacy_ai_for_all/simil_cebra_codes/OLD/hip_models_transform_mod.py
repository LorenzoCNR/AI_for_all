
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
if len(sys.argv) < 2:
    print("Too few args!!!")

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
def save_manif_data(hdf5_path, manif, labels=None, include_labels=False, group_name="TransformedData"):
    with h5py.File(hdf5_path, 'a') as hdf:  # 'a' for append or create if not exist
        if group_name not in hdf:
            group = hdf.create_group(group_name)
        else:
            group = hdf[group_name]

        dataset_name = 'transformed_data'
        labels_name = 'labels'

        # Update or create the transformed data dataset
        if dataset_name in group:
            del group[dataset_name]  # Remove old data if exists
        group.create_dataset(dataset_name, data=manif, compression='gzip', compression_opts=9)
        print(f"Transformed data saved under '{group_name}/{dataset_name}'.")

        # Update or create the labels dataset if required
        if include_labels and labels is not None:
            if labels_name in group:
                del group[labels_name]  # Remove old labels if exists
            group.create_dataset(labels_name, data=labels)
            print(f"Labels included under '{group_name}/{labels_name}'.")

        # Update metadata with the current time
        hdf.attrs['last_modified'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def run_hip_models_transform(hdf5_path, external_data_path=None,include_labels=False):
    with h5py.File(hdf5_path, 'r+') as hdf:
        # Load model
        model_input_path = hdf.attrs['model_input_path']
        model_fit = jl.load(model_input_path)

        # load data
        if external_data_path:
            ### 
            data = np.load(external_data_path)  
        else:
            # default data are neural data
            data_path = hdf.attrs['data_to_transform']
            labels_path = hdf.attrs['labels_']

            data = hdf[data_path][:]    
            labels=hdf[labels_path][:]

        seed = hdf.attrs['seed']
        set_seeds(seed)

        # 
        transformed_data = model_fit.transform(data)

        # Output management
        output_folder = Path(hdf.attrs['output_folder'])
        output_hdf5_path = output_folder / "transformed_data.hdf5"

        # Saved transformed data
        save_manif_data(output_hdf5_path, transformed_data, labels=labels if include_labels else None, include_labels=include_labels)
        print(f"Transformed data saved in {output_hdf5_path}")

       

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_hdf5> [external_data_path]")
        sys.exit(1)
    
    hdf5_path = sys.argv[1]
    external_data_path = sys.argv[2] if len(sys.argv) > 2 else None
    run_hip_models_transform(hdf5_path, external_data_path, include_labels=True)



   