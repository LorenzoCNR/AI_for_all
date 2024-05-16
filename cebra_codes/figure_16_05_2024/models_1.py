#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 00:41:33 2024

@author: zlollo
"""


#!pip install --pre 'cebra[datasets,demos]'
#!pip install ripser
import os
import sys
import warnings
import typing
os.chdir(os.getcwd())
os.getcwd()

#### windows
#base_dir=r'F:\CNR_neuroscience\cebra_git'
## ubuntu
#base_dir=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git'

os.chdir(base_dir)

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


### UBUNTU
# sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party')
# sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party/pivae')

# ### WINDOWS
# # Correct the paths by removing the leading backslash if you're specifying an absolute path
# #sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party')
# #sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party\\pivae')


# import pivae_code.datasets 
# import pivae_code.conv_pi_vae
# import pivae_code.pi_vae



off_r = 5
off_l = 5
split_no=0
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


### Load Parameters from YAML file

def load_params(params_path):
    with open(params_path, 'r') as file:
        params = yaml.safe_load(file)
    return params
    
    
### Inirtialize the model
    
def create_model(model_type, params):
    if model_type == 'tsne':
        return openTSNE.TSNE(**params)
    elif model_type == 'umap':
        return umap.UMAP(**params)
    elif model_type in ['cebra_time', 'cebra_behavior','cebra_hybrid']:
        return CEBRA(**params)
    else:
        raise ValueError("Unsupported model type")


## Run the initialized model 
def run_model(model_type, params, X, y):
    model = create_model(model_type, params)

    if model_type in ['cebra_time', 'umap', 'tsne']:
        model= create_model(model_type, params)
        fitted_model=model.fit(X)
        embeddings = fitted_model.transform(X)
        #save_results(h5_file, 'Cebra', 'Cebra_Time', config, embeddings)
    elif model_type in ['cebra_behavior', 'cebra_hybrid']:
        model= create_model(model_type, params)
        fitted_model=model.fit(X,y)
        embeddings = fitted_model.transform(X)
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

   
def main(input_dir, output_dir, rat, hdf5_name, param_file, model_type, use_grid, replace):
    params = load_params(param_file)[model_type]
    fixed_params = params['fixed']
    grid_params = params.get('grid', {})
    
    param_list = list(ParameterGrid(grid_params)) if use_grid else [next(iter(ParameterGrid(grid_params)))]
    
    data_ = load_dataset_for_rat(rat, split_no=0)
    X = data_.neural.numpy()
    y = data_.continuous_index.numpy()
    
    for p in param_list:
        model_params = {**fixed_params, **p}
        embeddings = run_model(model_type, model_params, X, y)
        save_results(os.path.join(output_dir, hdf5_name), model_type, p, embeddings, replace)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters for the model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')

    args = parser.parse_args()

    # Caricare i parametri dal file di configurazione
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    input_dir = config['input_dir']
    output_dir = config['output_dir']
    rat_name = config['rat_name']
    params_file = config['params_file']
    path_to_yaml = os.path.join(input_dir, params_file)
    output_file = config['output_file']
    path_to_output = os.path.join(output_dir, output_file)
    model_type = config['model_type']
    use_grid = config['use_grid']
    replace = config['replace']

    main(input_dir, output_dir, rat_name, path_to_output, path_to_yaml, model_type, use_grid, replace)
    
    
    
    # parser = argparse.ArgumentParser(description='Process some parameters for the model.')
    # parser.add_argument('--input_dir', type=str, required=True, help='Input directory path')
    # parser.add_argument('--output_dir', type=str, required=True, help='Output directory path')
    # parser.add_argument('--rat', type=str, required=True, help='Name of the rat')
    # parser.add_argument('--hdf5_name', type=str, required=True, help='HDF5 file name for output')
    # parser.add_argument('--param_file', type=str, required=True, help='Parameter file in YAML format')
    # parser.add_argument('--model_type', type=str, required=True, choices=['tsne', 'umap', 'cebra_time', 'cebra_behavior', 'cebra_hybrid'], help='Type of model to use')
    # parser.add_argument('--use_grid', type=bool, default=True, help='Whether to use grid search for parameters')
    # parser.add_argument('--replace', type=bool, default=False, help='Whether to replace existing datasets')
    
    # args = parser.parse_args()
    
    # input_dir = r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'
    # output_dir = r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'
    # rat_name = 'achilles'
    # params_file = 'model_params_1.yaml'
    # path_to_yaml = os.path.join(input_dir, params_file)
    # output_file = 'manifold_1.hdf5'
    # path_to_output = os.path.join(input_dir, output_file)
    # model_type = 'cebra_hybrid'
    # use_grid = True
    # replace = True
    # main(input_dir, output_dir, rat_name, path_to_output, path_to_yaml, 
    #      model_type, use_grid,replace)

        # parser = argparse.ArgumentParser(description="Run models and save embeddings.")
    # parser.add_argument("input_dir", type=str, help="Directory containing input data files")
    # parser.add_argument("output_dir", type=str, help="Directory to save the HDF5 results file")
    # parser.add_argument("rat", type=str, help="Identifier of the rat")
    # parser.add_argument("hdf5_name", type=str, help="HDF5 file name for storing results")
    # parser.add_argument("param_file", type=str, help="Path to the YAML parameters file")
    # parser.add_argument("model_type", type=str, choices=['cebra_time', 'cebra_behavior', 'tsne', 'umap', 'conv_pivae'], help="Type of model to run")
    # parser.add_argument("--use_grid", action="store_true", help="Whether to perform grid search")
    
    # args = parser.parse_args()

    # params = load_params(args.param_file)
    # fixed_params = params['fixed']
    # grid_params = params.get('grid', {})

#main(args.input_dir, args.output_dir, args.rat, args.hdf5_name, args.param_file, args.model_type, args.use_grid)

# Just print names in the command #

    
    
    


    
















