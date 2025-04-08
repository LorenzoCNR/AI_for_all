#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:07:54 2024

@author: zlollo
"""
import sys
import argparse
import yaml
import logging
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import openTSNE
import scipy.sparse as sp
import sympy
from joblib import Parallel, delayed
#import ripser
import numpy as np
import torch
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import umap
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from scipy import stats
# import tensorflow as tf
from cebra import CEBRA
#from dataset import SingleRatDataset  
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
#import inspect
#import torch
from cebra.datasets.hippocampus import *
import h5py
import pathlib
from matplotlib.markers import MarkerStyle
import seaborn as sns
from scipy.sparse import lil_matrix
import datetime
from sklearn.model_selection import ParameterGrid
from keras.callbacks import ModelCheckpoint
from keras import backend as K

### UBUNTU
sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party')
sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party/pivae')


# ### WINDOWS
#Correct the paths by removing the leading backslash if you're specifying an absolute path
#sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party')
#sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party\\pivae')

#import pivae_code.datasets 
#import pivae_code.conv_pi_vae
#import pivae_code.pi_vae


off_r = 5
off_l = 5

def load_dataset_for_rat(rat, split_no, split_type='all', seed=None):
    """
    Args:
        rat (str)
        split_no (int)
        split_type (str): Tipo di split, default è 'all'.
        seed (int): optional seed; default none. 

    """
    dataset_name = f'rat-hippocampus-{rat}-3fold-trial-split-{split_no}'
    dataset = cebra.datasets.init(dataset_name, split=split_type)

    # Configure offset 
    if hasattr(dataset, 'offset'):
        dataset.offset.right = off_r
        dataset.offset.left = off_l

    return dataset

###### split data in batches to make em more derivable (per conv_pivae)

def custom_data_generator(x_all, u_one_hot):
    while True:
        for ii in range(len(x_all)):
            yield ([x_all[ii], u_one_hot[ii]], None)


def make_loader(X,y, batch_size):
    def _to_batch_list(x, y, batch_size):
        if x is not None and y is not None:
            x = x.squeeze()
            if len(x.shape) == 3:
                x = x.transpose(0,2,1)
            x_batch_list = np.array_split(x, int(len(x) / batch_size))
            y_batch_list = np.array_split(y, int(len(y) / batch_size))
        else:
            return None, None
       
        return x_batch_list, y_batch_list
    
    x_batches, y_batches = _to_batch_list(X, y, batch_size)

    # if x_batches is None or y_batches is None:
    #    return None, None, None

    loader = custom_data_generator(x_batches, y_batches)
    return x_batches, y_batches, loader



def load_params(params_path):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params
    
    
### Inirtialize the model
    
def create_model(model_type, params,**extra_params):
    if model_type == 'tsne':
        return openTSNE.TSNE(**params)
    elif model_type == 'umap':
        return umap.UMAP(**params)
    elif model_type in ['cebra_time', 'cebra_behavior','cebra_hybrid']:
        return CEBRA(**params)
    elif model_type == 'conv_pivae':
        all_params = {**params, **extra_params}
        return pivae_code.conv_pi_vae.conv_vae_mdl(**all_params)
    
    else:
        raise ValueError("Unsupported model type")


## Run the initialized model 
def run_model(model_type, params, data_):
    X = data_.neural.numpy()
    y = data_.continuous_index.numpy()
   
    extra_params = {}
   
    if model_type == 'conv_pivae':
       # Model dependent parameters of conv_pivae
       extra_params['dim_x'] = X.shape[1]
       extra_params['dim_u'] = 3  #
       extra_params['time_window'] = 10 
       fit_params = params.get('fit_params', {})
       ### pre defined parameters
       batch_size = params.get('fit_params', {}).get('batch_size', 200)
       epochs = params.get('fit_params', {}).get('epochs', 1000)
       verbose = params.get('fit_params', {}).get('verbose', 1)


       
       ## da capire, poco senso...transform=predict?
     
    model = create_model(model_type, params, **extra_params)
       
    if model_type in ['cebra_time', 'umap', 'tsne']:     
        fitted_model=model.fit(X)
        embeddings = fitted_model.transform(X)
        
        #save_results(h5_file, 'Cebra', 'Cebra_Time', config, embeddings)
    elif model_type in ['cebra_behavior', 'cebra_hybrid']:        
        fitted_model=model.fit(X,y)
        embeddings = fitted_model.transform(X)
    elif model_type == 'conv_pivae':
           # Model dependent parameters of conv_pivae
           X_,y_ = data_[torch.arange(len(data_))].numpy(), data_.index.numpy()
           X_l ,y_l, loader_l= make_loader(X_,y_, batch_size)
           fitted_model=model.fit(x=loader_l,
                         steps_per_epoch=len(X_l), epochs=epochs,
                         verbose=verbose)
                         #validation_data=valid_loader,
                         #validation_steps=validation_steps,
                         #callbacks=[mcp])
           outputs=model.predict([np.concatenate(X_l),
                                np.concatenate(y_l)])
           
           embeddings=outputs[6]

    return embeddings

'''

## store results in hd5: hierarchically:
    - Create database if not existing (call from main)
    - Create GRoup if not existing (actually group name is model type within db)
    - Save embedding/manifold in the given group according to time and grid params
    
'''

def save_results(h5_file, model_group, params, embeddings,replace=False):
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    param_name = '_'.join(f"{k}{v}" for k, v in sorted(params.items()))
    dataset_name = f"{model_group}_{current_date}_{param_name}"
    with h5py.File(h5_file, "a") as h5f:
        if model_group not in h5f:
                group = h5f.create_group(model_group)
        else:
                group = h5f[model_group]
            
        if dataset_name in group:
            if replace:
                del group[dataset_name]
                dataset = group.create_dataset(dataset_name, data=embeddings)
                dataset.attrs['params'] = yaml.dump(params)
                print(f"Dataset {dataset_name} replaced.")
            else:
                print(f"Dataset {dataset_name} already exists, skipping save.")
        else:
            dataset = group.create_dataset(dataset_name, data=embeddings)
            dataset.attrs['params'] = yaml.dump(params)
            print(f"Dataset {dataset_name} saved.")

   
def main(input_dir, output_dir, rat_, hdf5_name, param, model_type, use_grid, replace):
    path_to_yaml = os.path.join(input_dir, param)
    params = load_params(path_to_yaml)[model_type]
    fixed_params = params['fixed']
    grid_params = params.get('grid', {})
    
    param_list = list(ParameterGrid(grid_params)) if use_grid else [next(iter(ParameterGrid(grid_params)))]
    data_ = load_dataset_for_rat(rat_, split_no=0)
  
    
    for p in param_list:
        model_params = {**fixed_params, **p}
        embeddings = run_model(model_type, model_params, data_)
        save_results(os.path.join(output_dir, hdf5_name), model_type, p, embeddings, replace)

if __name__ == "__main__":
    ## Ubuntu
    def_input_dir = r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes/figure_16_05_2024'
    def_output_dir = r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes/figure_16_05_2024'
   
    # Windows
    #def_input_dir = r'F:\CNR_neuroscience\cebra_git\Cebra_for_all\cebra_codes\figure_16_05_2024'
    #def_output_dir = r'F:\CNR_neuroscience\cebra_git\Cebra_for_all\cebra_codes\figure_16_05_2024'
    default_rat = 'achilles'
    default_hdf5_name = 'manifold_3.hdf5'
    default_param = 'model_params_1.yaml'
    default_model_type = 'tsne'
    default_use_grid = False
    default_replace = True

    #main(input_dir, output_dir, rat_, path_to_output, path_to_yaml, 
     #     model_type, use_grid,replace)
    
    parser = argparse.ArgumentParser(description='Process some parameters for the model.')
    #parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')



    # main(input_dir, output_dir, rat_name, path_to_output, path_to_yaml, model_type, use_grid, replace)
    
    
    parser = argparse.ArgumentParser(description='Process some parameters for the model.')
    parser.add_argument('--input_dir', type=str, default=def_input_dir, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default=def_output_dir, help='Output directory path')
    parser.add_argument('--rat', type=str, default=default_rat, help='Name of the rat')
    parser.add_argument('--hdf5_name', type=str, default=default_hdf5_name, help='HDF5 file name for output')
    parser.add_argument('--param', type=str, default=default_param, help='Parameter file in YAML format')
    parser.add_argument('--model_type', type=str, default=default_model_type, help='Type of model to use')
    parser.add_argument('--use_grid', action='store_true', default=default_use_grid, help='Whether to use grid search for parameters')
    parser.add_argument('--replace', action='store_true', default=default_replace, help='Whether to replace existing datasets')

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.rat, args.hdf5_name, args.param, args.model_type, args.use_grid, args.replace)


#main(args.input_dir, args.output_dir, args.rat, args.hdf5_name, args.param_file, args.model_type, args.use_grid)

# Just print names in the command #

#f_name=path_to_output

#pd.read_hdf(f_name, key='cebra_time')
#
# #### print group and datasets name
# def print_names(name, obj):
#     print(name)
#     if isinstance(obj, h5py.Group):
#         print(f"{name} is a group")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"{name} is a dataset")
#     else:
#         print(f"{name} Unknown type")
  
# f_name=path_to_output
# with h5py.File(f_name, 'r') as file:
#     file.visititems(print_names)

    


















