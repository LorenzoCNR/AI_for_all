# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 12:55:03 2024

@author: zlollo2
"""
'''
ARXIV Arxiv_161820 Lognibeni mail sapienza
Scholar...mail sapienza



'''

import os



#os.getcwd()

import sys
import warnings
import typing
import numpy as np
import yaml
import argparse
import re
import joblib as jl
import logging
import time
import openTSNE
import argparse
import scipy.sparse as sp
import sympy
from sklearn.model_selection import ParameterGrid, train_test_split
from joblib import Parallel, delayed
import numpy as np
import torch
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import umap
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from cebra import CEBRA
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from cebra.datasets.hippocampus import *
import h5py
import pathlib
from matplotlib.markers import MarkerStyle
import seaborn as sns
import pickle
import joblib
import copy
from sklearn.model_selection import train_test_split

#from sampling_infonce import *
from h5_management import *
from split_data_fncts import *
from knn_decoder import *
from model_utils import *
#from model_utils import run_model
print('all modules  are on')
#from process_utils import *
# #################################### LOAD DATA ##################################


def load_datasets(file_path, rat_keys):
    """
     LOAD and convert data in dict of dicts

    """
    dataset = jl.load(file_path)
    if len(dataset) != len(rat_keys):
        raise ValueError("La lunghezza delle chiavi non corrisponde al dataset.")
    
    datasets = {key: diz for key, diz in zip(rat_keys, dataset)}
    return datasets


def split_datasets(datasets, train_ratio, val_ratio,
                   subtrain=False, shuffle=False, seed=42):
    """
    Split data in train test valid (eventually subtrain).
    """
    return split_data(datasets,subtrain=subtrain, shuffle=shuffle, seed=seed,
                      train_ratio=train_ratio, val_ratio=val_ratio)



#
##################################### SPLIT DATA  ###############################

extra_params = {}
def load_params(params_path):
    # check existence
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"file not found{params_path}")
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(f"Error in reading from yaml: {exc}")
        return None  
    #except Exception as exc:
     #   print(f"Error!! {exc}")
     #   return None

    return params

###################################### DATA PROCESS #############################
def process_data(data_, param_list, fixed_params, hdf5_path, model_type,
                 rat_name, use_grid, batch_size,subtrain=False):
    """
    Processes data, creates models, and runs KNN decoding based on 
    subtrain and validation data.
    """
    best_error = float('inf')

    for p in param_list:
        model_params = {**fixed_params, **p}

        #
        results = run_model(model_type, model_params, data_, subtrain=False)

        if results[0] is None:
            continue

        embeddings_train, embeddings_valid, embeddings_sub_train, model, y_train, y_valid, y_sub_train, X_sub_train = results

        # FInd best k for KNN
        print("Finding best k using internal subtrain and validation sets...")
        best_k, val_score, val_pos_err, val_pos_score = find_best_k(embeddings_sub_train, embeddings_valid, y_sub_train, y_valid)
        print(f"Best k is: {best_k} with validation score: {val_score}")
        
        # set decoders given the best k 
        print("Setting up decoders with best k...")
        pos_decoder = KNeighborsRegressor(n_neighbors=best_k, metric='cosine')
        dir_decoder = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
        pos_decoder.fit(embeddings_sub_train, y_sub_train[:, 0])
        dir_decoder.fit(embeddings_sub_train, y_sub_train[:, 1])
        
        # PErfroamcne evaluation on validation set
        print("Evaluating performance on validation set...")
        valid_score, valid_pos_err, valid_pos_score, predictions = decoding_mod(pos_decoder, dir_decoder, embeddings_valid, y_valid)
        
        if valid_pos_err < best_error:
            best_error = valid_pos_err
            print(f"Updated best error: {best_error}")
            group_name = f"{rat_name}_best_model_knn"
            path, group = save_model_to_hdf5(model, model_params, hdf5_path, group_name, best_error)
            data_[f'best_knn_model_hdf5_{model_type}'] = (path, group)
            data_[f'model_params_knn_{model_type}'] = model_params
            data_[f'embeddings_train_knn_{model_type}'] = embeddings_train
            data_[f'embeddings_sub_train_knn_{model_type}'] = embeddings_sub_train
            data_[f'embeddings_valid_knn_{model_type}'] = embeddings_valid
    
    print("Model and data processing completed for:", rat_name)

##############
def main(input_dir, output_dir, param, model_type,rat_list, use_grid, replace,  subtrain):    ### define subject keys and load data
    rat_keys = ['achilles', 'cicero', 'gatsby', 'buddy']
    datasets_path = os.path.join(input_dir, 'd_rats.joblib')
    datasets = load_datasets(datasets_path, rat_keys)
    
    ## Split data
   
    datasets_split = split_datasets(datasets, train_ratio=0.7, val_ratio=0.15, subtrain=subtrain, shuffle=False, seed=42)
    
    
    ### Parameters Management
    path_to_yaml = os.path.join(input_dir, param)
    params = load_params(path_to_yaml)
    fixed_params = params[model_type]['fixed']
    grid_params = params[model_type].get('grid', {})
    batch_size = fixed_params.get('batch_size', 32)
    
    ## If use_grid is True, rotate (grid) parameters, otherwise use first grid param
    if use_grid:
        param_list = list(ParameterGrid(grid_params))
    else:
        first_grid_params = {k: v[0] for k, v in grid_params.items()}
        param_list = [{**fixed_params, **first_grid_params}]
    
    ### Check if input is single or dictionary of dictionaries
    if len(datasets_split) == 1:
        # 
        rat_name = list(datasets_split.keys())[0]
        data_ = datasets_split[rat_name]
        hdf5_path = os.path.join(output_dir, f"{rat_name}_{model_type}_models.hdf5")
        process_data(data_, param_list, fixed_params, hdf5_path, model_type, rat_name, use_grid, batch_size)
    
     ### Check if processing multiple rats based on the rat list
    elif rat_list:
         selected_data = {k: datasets_split[k] for k in rat_list if k in datasets_split}
         for rat_name, data_ in selected_data.items():
            hdf5_path = os.path.join(output_dir, f"{rat_name}_{model_type}_models.hdf5")
            process_data(data_, param_list, fixed_params, hdf5_path, model_type, rat_name, use_grid, batch_size)
    
    ### Process all rats if no rat list is provided
    else:
        for rat_name, data_ in datasets_split.items():
            hdf5_path = os.path.join(output_dir, f"{rat_name}_{model_type}_models.hdf5")
            process_data(data_, param_list, fixed_params, hdf5_path, model_type, rat_name, use_grid, batch_size)

if __name__ == "__main__":
    input_dir = os.getcwd()
    output_dir = input_dir

    default_param = 'model_params_1.yaml'
    default_model_type = 'cebra_behavior'
    default_use_grid = True
    default_replace = True
    default_subtrain = False

    parser = argparse.ArgumentParser(description='Process some parameters for the model.')
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Output directory path')
    parser.add_argument('--param', type=str, default=default_param, help='Parameter file in YAML format')
    parser.add_argument('--model_type', type=str, default=default_model_type, help='Type of model to use')
    parser.add_argument('--rat_list', nargs='+', help='List of subject keys to process')
    parser.add_argument('--use_grid', action='store_true', default=default_use_grid, help='Whether to use grid search for parameters')
    parser.add_argument('--replace', action='store_true', default=default_replace, help='Whether to replace existing datasets')
    parser.add_argument('--subtrain', action='store_true', default=default_subtrain, help='Whether to use subtrain in the dataset split')
    
    args = parser.parse_args()
    print(f"subtrain is set to: {args.subtrain}")  
    rat_list=['achilles']
    subtrain=False
    
    main(args.input_dir, args.output_dir, args.param, args.model_type,rat_list,  args.use_grid, args.replace,subtrain)