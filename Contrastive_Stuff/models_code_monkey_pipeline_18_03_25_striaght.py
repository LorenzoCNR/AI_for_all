# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 10:37:47 2025

@author: loren
"""

import os
import sys
import math
from pathlib import Path
import argparse
import logging
import mimetypes
from statsmodels.tsa.seasonal import seasonal_decompose

### load path (use the function in module some functions)
'''
Assume a path structure:

main_folder
|___ data(folder)
|     |__project_data_folder1
                    |__ dati_cebra.jl (file dati)
      |__project_data_folder2
                    |__ dati_mirco.mat (file dati mat file, jle file etc)
|___ project_root(folder)
            |__d_cod_mon_Mirco.py
            |__some_functions.py
            |___EEG_ANN_pipeline(folder)
                    |__data (folder)
                    |__helpers (folder)
                    |__etc....
            |___output directory (folder)
            

'''
# need to declare:
# 1) PROJECT root directory
#  windows directories
# i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

# #  ubuntu directories
# #i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

# os.chdir(i_dir)
# os.getcwd()
# from some_functions import *
# import json
# import copy
# import time
# import numpy as np
# import yaml
# import pickle
# import warnings
# import logging
# #import umap 
# import openTSNE
# import random
# import typing
# import joblib as jl
# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from torch import nn
# from torch.utils.data import DataLoader
# from sklearn.datasets import load_digits
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# from sklearn.model_selection import ParameterGrid, train_test_split
# from sklearn.model_selection import ParameterGrid, ParameterSampler, RandomizedSearchCV

# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# import sklearn.metrics
# from scipy import stats
# import seaborn as sns
# from matplotlib.collections import LineCollection
# from matplotlib.markers import MarkerStyle
# from joblib import Parallel, delayed
# import torch
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
# import cebra.datasets
# from cebra import CEBRA
# from cebra  import *
# from scipy import optimize as opt

# from cebra.datasets.hippocampus import *
#  # Sostituisci con una funzione specifica presente in some_functions.py
# from model_utils import *
# # Random Seeds
# torch.manual_seed(42)
# random.seed(42)
# np.random.seed(42)

# # Config GPU
# torch.cuda.manual_seed(42)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

# def prompt_for_paths_and_settings():
    
#     print("Project Configuration:")

#     data_dir = input("Insert Project main directory (default: 'data'): ") or "data"
#     sub_data = input("Insert Project Specific directory (default: 'Monkeys_Mirco'): ") or "Monkeys_Mirco"
#     pipe_path = input("Insert Pipeline folder (default: 'EEG-ANN-Pipeline'): ") or "EEG-ANN-Pipeline"
#     out_dir = input("Inserisci la cartella di output (default: 'contrastive_output'): ") or "contrastive_output"

#     # Set directories with given paramaeters
#     project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths(
#         data_dir=data_dir, sub_data=sub_data, out_dir=out_dir, pipe_path=pipe_path, change_dir=False)
    
#     print(f"Project Root: {project_root}")
#     print(f"EEG Pipeline Path: {eeg_pipeline_path}")
#     print(f"Default Input Directory: {default_input_dir}")
#     print(f"Default Output Directory: {default_output_dir}")
    
#     return project_root, eeg_pipeline_path, default_output_dir, default_input_dir

# # load data 
# def load_dataset(default_input_dir, dataset_name, data_format="mat"):
#     # 
#     data = load_data(default_input_dir, dataset_name, data_format)
#     return data


# ### likely resampling 


# # 1. Prompt main directories
# project_root, eeg_pipeline_path, default_output_dir, default_input_dir = prompt_for_paths_and_settings()

# # 2. Load data
# dataset_name = input("Inserisci il nome del dataset (default: 'dati_mirco_18_03_joint'): ") or "dati_mirco_18_03_joint"
# data = load_dataset(default_input_dir, dataset_name)

def main():
    output_folder=default_output_dir
    # Get X, y and trial id
    X = data['s_passive_neural']
    y_dir = data['s_passive_trial']
    trial_id = data['s_passive_trial_id']
    trial_id=trial_id.flatten()
    original_label_order = np.sort(np.unique(y_dir))
    
    # Generate trial lists
    c_t = np.concatenate([[0], np.where(np.diff(trial_id) != 0)[0] + 1, [len(y_dir)]], dtype=int)
    c_t_list = [(c_t[i], c_t[i + 1] - 1) for i in range(len(c_t) - 1)]
    trial_len=np.diff(c_t)
    trial_length=trial_len[0]
    const_len = np.var(trial_len) == 0
    #  Resampling (on demand)
    resample = input("Vuoi eseguire un resampling? (yes/no) [no]: ").strip().lower() in ["yes", "y"]
    if resample:
        varnames = input("Inserisci nomi variabili separati da virgola (default: X, y_dir): ") or "X, y_dir"
        available_vars = {'X': X, 'y_dir': y_dir, 'trial_id': trial_id}
        l_data = [available_vars[var.strip()] for var in varnames.split(',')]
        step = int(input("Inserisci la dimensione della finestra di resampling (default: 10): ") or 10)
        overlap = int(input("Inserisci il numero di punti di sovrapposizione (default: 5): ") or 5)
        mode = input("Modo di resampling ('overlapping' o 'disjoint') [default: overlapping]: ") or "overlapping"
        method_list = input("Metodi di resampling (default: mean, center): ").split(",") or ["mean", "center"]
        methods = {i: m.strip() for i, m in enumerate(method_list if method_list != [''] else ["mean", "center"])}
        #methods = {0: "mean", 1: "center"}
    
        resampled_data, r_trial_lengths, r_trial_indices = f_resample(
            l_data, c_t_list, step, overlap,methods, mode=mode, normalization=True
        )
        X = resampled_data[0]
        y_dir = resampled_data[1]
    
        # Nuovo c_t da ricalcolare
        r_trial = r_trial_indices[0]
        start_points = [start for (start, _) in r_trial]
        start_points.append(r_trial[-1][1] + 1)
        c_t = np.array(start_points, dtype=int)
        trial_len = np.diff(c_t)
        trial_length=trial_len[0]
        const_len = np.var(trial_len) == 0
    else:
        pass
        # X = X
        # y_dir = y_dir
    
    y_dir_original = y_dir.flatten()  
    swap_dict=None
    #Swapping (on demand)
    do_swap = input("Vuoi eseguire lo swap delle etichette? (yes/no) [no]: ").strip().lower() in ["yes", "y"]
    if do_swap:
        print("Etichette uniche prima dello swap:", np.unique(y_dir))
        swap_dict = eval(input("Inserisci dizionario di swap (es. {3:6, 6:3}): "))
        y_dir_original=y_dir.flatten()
        y_dir = swap_labels(y_dir_original, swap_dict)
        print("Etichette dopo lo swap:", np.unique(y_dir))
        
    # Set model data
    X_=X
    y_dir_=y_dir.flatten()
    data_={"X":X_,"y":y_dir_}
    train_data_=['X', 'y']
    transform_data_=['X']
    print(trial_length)
   #from cebra.sklearn.metrics import infonce_loss
    param_file='model_params_1.yaml'
    params_path = Path(project_root) / param_file
    params = load_params(params_path)
    
    ## define the model and declare the data u want to use
    ### model type: cebra_time, cebra_behavior, cebra_hybrid, UMAP, TSNE, covn_pivae
    
    model_type = input("Modello da usare (cebra_time, cebra_behavior, etc.) [default: cebra_behavior]: ") or 'cebra_behavior'
    
    fixed_params = params[model_type]['fixed']
    grid_params = params[model_type].get('grid', {})
    use_grid=False
    ### fixed require only train data
    ### random and grid search want also validation data
    ### run the model and store the results
    ### mode fisso grid search o random 
    mode='fixed'
    if mode == "fixed":
        param_list = [{**fixed_params}]
    elif mode == "grid_search":
        param_list = list(ParameterGrid(grid_params))
    elif mode == "random_search":
        random_search = ParameterSampler(grid_params, n_iter=10, random_state=42)
        param_list = list(random_search)
    else:
        raise ValueError("Invalid mode. Choose 'fixed', 'grid_search', or 'random_search'.")
    for p in param_list:
        print(f"\n run model {model_type} with parameters: {p}")
        model_params = {**fixed_params, **p}
    
    
    # y_dir_b=y_dir.reshape(len(y_dir),1)
    # yy=np.concatenate((y_pos,y_dir_b),axis=1)
    # yy=y_dir
    #validation_X=data_split['X_val']
    #validation_y=data_split['y_val']
    #data_split[train_data[0]]
    
    '''
    - train data is a list of strings pointing to the data_ dict's':
             -neural data
              (X matrix, Time*Chns)
             - Behaviorial data y (labels)
               either
               discrete labels
               continuous label
               uni or multidimensional (?)
               single or multi labels        
             
    '''
    
    
    ###define the trajectories. First monkey is 1-8, second is 9-16
    # first
    #n_traj=np.arange(8)
    for p in param_list:
         print(p)
         
         print(f"\n run model {model_type} with parameters: {p}")
         model_params = {**fixed_params, **p}
         l_r=model_params['learning_rate']
         n_h_u=model_params['num_hidden_units']
         temp=model_params['temperature']
         default_title = f"S_CEBRA_cond3_resampled_shift5_sum_lr_{l_r}_nhu_{n_h_u}_temp{temp}.html"
         title_ = input(f"Inserisci il nome del file di output (default: {default_title}): ") or default_title
             
                 #print(type(model_params))
         # Esegui `run_model` con i dati di input e i parametri specifici
         results = run_model(
             model_type=model_type,
             params=model_params,
             data_=data_,
             train_data=train_data_,
             #transform_data=transform_data_,
             save_results=False
             
                 )
         results['param']=p
         results['train final loss']=results['fitted_model'].state_dict_['loss'][-1].numpy()
         results['train loss']=results['fitted_model'].state_dict_['loss'].numpy()
         
         
         fitted_model=results['fitted_model']
         X_hat=fitted_model.transform(X_)
         c_s="maroon"
         plot_direction_averaged_embedding(
                X_hat,
                y_dir_original,
                original_label_order,
                c_s,
                output_folder,
                title_,
                trial_length,
                constant_length=const_len,
                ww=0,
                label_swap_info=swap_dict
            ) 
main()

