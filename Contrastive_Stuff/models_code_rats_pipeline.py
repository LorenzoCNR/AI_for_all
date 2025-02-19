# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 00:16:57 2025

@author: loren
"""

import os
import sys
from pathlib import Path
### directory su windows
i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

## directories su ubuntu
#i_dir=r'/media/lorenzo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

os.chdir(i_dir)
os.getcwd()

import argparse
import json
import copy
import time
import yaml
import pickle
import warnings
import logging
import random
import typing

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.model_selection import ParameterGrid, ParameterSampler, RandomizedSearchCV

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from scipy import stats
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
from joblib import Parallel, delayed
import torch
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import cebra.datasets
from cebra import CEBRA
from cebra  import *

from cebra.datasets.hippocampus import *
 # Sostituisci con una funzione specifica presente in some_functions.py
from some_functions import *
from model_utils import *


torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

######################## Setto le paths


def setup_paths():
    """
   Dynamic path config
    """
   # If __file__ is not available, use the current working directory (cwd).
    try:
        project_root = Path(__file__).resolve().parent
    except NameError:
        project_root = Path(os.getcwd())  # Fallback to cwd if __file__ is not available
    
    print(f"Project root is: {project_root}")
   
    default_input_dir= project_root.parent/ "data" / "rat_hippocampus"
    print(f"data_path: {default_input_dir}")
    #  "contrastive_output" (output directory)
    default_output_dir = project_root / "contrastive_output"
    print(f"ouptut path: {default_output_dir}")

   
    # Build output dir if not existing
    default_output_dir.mkdir(exist_ok=True)

    #os.chdir(project_root)

    
    return project_root, default_output_dir, default_input_dir


#### PArto dai dati 
### load data....inspect data...
# function on purpose of inspecting data
#### gli devi passare la directory di input
project_root, default_output_dir, default_input_dir = setup_paths()
input_dir=default_input_dir
output_dir=default_output_dir
name='achilles'






path = Path(input_dir) / f"{name}.jl"
##### the data imported must be in the form of a dictionart
data = jl.load(path)
### define X  y  (label are not necessarily defined) and trials
### Identify what are the neural (spikes or whatever) and behavioural data
data['X']=data['spikes']
data['y']=data['position']
#### define the samplign frequency
fs=40
### create trials with ad hoc function.. in some function
# c_T counter and time colum
data['trials'], c_t = create_trial_ids(data['position'][:,1])
data['time_sec']=np.arange(0,len(data['spikes']))/fs
print(len(c_t))


######################## Pre process of data #############################
### to do (what and how?)

################# Data Split
### split mode 
# case 1 Train, Validation, and Test
# case 2 Train and Validation only 
# case 3 Train (Subtrain and Validation within Train) and Test
case=1
train_ratio=0.80
val_ratio=0.10

data_split = split_data_trials(data, case=case, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=False, seed=42, verbose=True)

print("keys in data_split:", data_split.keys())



#################################### MODEL PART ##############################
### point to the (hyper)parameters file
### params file is a yaml
from some_functions import *
from model_utils import *
#from cebra.sklearn.metrics import infonce_loss
param_file='model_params_1.yaml'
params_path = Path(project_root) / param_file
params = load_params(params_path)

## define the model and declare the data u want to use
### model type: cebra_time, cebra_behavior, cebra_hybrid, UMAP, TSNE, covn_pivae

model_type='cebra_behavior'

fixed_params = params[model_type]['fixed']
grid_params = params[model_type].get('grid', {})
use_grid=False
### fixed require only train data
### random and grid search want also validation data
### run the model and store the results
### mode fisso grid search o random 
mode='random_search'
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

train_data=['X','y']
validation_X=data_split['X_val']
validation_y=data_split['y_val']
data_split[train_data[0]]

transform_data=[]
### run and validate model (store loss del train...facile) ma se voglio


results_list=[]
for p in param_list:
     print(p)
     
     print(f"\n run model {model_type} with parameters: {p}")
     model_params = {**fixed_params, **p}

     # Esegui `run_model` con i dati di input e i parametri specifici
     results = run_model(
         model_type=model_type,
         params=model_params,
         data_=data_split,
         train_data=train_data,
         #transform_data=transform_data,
         save_results=False
         
             )
     results['param']=p
     results['train final loss']=results['fitted_model'].state_dict_['loss'][-1].numpy()
     results['train loss']=results['fitted_model'].state_dict_['loss'].numpy()
     ### compute infonce also on validation data: comment to skip it 
     ### refinement: provide the option with data
     results['valid loss'] = cebra.sklearn.metrics.infonce_loss(
         results['fitted_model'],
         validation_X,
         validation_y,
         num_batches=100,  # Numero di batch da considerare per la valutazione
         correct_by_batchsize=False,  # Correzione basata sulla batch size
     )
     
     results_list.append(results)
     
     min_val_loss=min(results_list, key=lambda d: d['valid loss'])
     min_train_loss=min(results_list, key=lambda d: d['train final loss'])
     
     
     if not results:
         continue



     # embedding_results[f"{model_type}_{str(p)}"] = {
     #     "model_params": model_params,
     #     **results
     # }

def run_pipeline(  model_type,use_grid,
                  mode, train_data, transform_data):



    

        embedding_results, data_split, temp, max_iters = process_data(data_split, [params], fixed_params, model_type)

        
    
        # Decoding with KNN
        results_dict = {}
        for config, results in embedding_results.items():
          z_train = results[f"embeddings_{train_data[0]}"]
          z_val = results[f"embeddings_{transform_data[0]}"]
          labels_train = data_split[train_data[1]]
          labels_val = data_split[transform_data[1]]
    
          knn_results = decoding_knn(z_train, z_val, labels_train, labels_val)

          results_dict[config] = {
              "params": results["model_params"],
              "test_score": knn_results[0],
              "pos_test_error": knn_results[1],
              "pos_test_error_perc": knn_results[2],
              "pos_test_score": knn_results[3],
          }

        #temperature = fixed_params.get('temperature', 'N/A')
        #max_iters = grid_params.get('max_iterations', 'N/A')
        title = f"{model_type} - {name} Train Embeddings | Temp={temp} | Iterations={max_iters}"
        plot_embs(z_train, labels_train,title)
        #plt.show() 
        #posdir_decode_CL = decoding_knn(z_train, z_val, labels_train, labels_val)
        #plot_embs(embedding_results['embedding_train'],data_split['y_train'])
    
        
        #input("\nPremi INVIO per continuare al prossimo set di parametri...")
        
    return {f"{name}_results_dict": results_dict}
    ####################### Function to laod parameteres #########################
    
  
 
if __name__ == "__main__":
    project_root, default_output_dir, default_input_dir = setup_paths()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    input_dir = default_input_dir
    output_dir = default_output_dir
    
    

    parser = argparse.ArgumentParser(description='Esegui la pipeline di elaborazione dati e embedding.')
    parser.add_argument("--mode", type=str, choices=["split", "train", "transform", "knn", "full"],
                        default="full", help="Modalità di esecuzione.")
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Output directory path')
    parser.add_argument('--param_file', type=str, default='model_params_1.yaml', help='Parameter file in YAML format')
    parser.add_argument('--model_type', type=str, default='cebra_behavior', help='Type of model to use')
    parser.add_argument('--name', type=str, default='achilles', help='Name of the subject (file name without extension)')
    parser.add_argument('--use_grid', type=bool, default=False, help='Whether to use grid search for parameters')
    parser.add_argument('--case', type=int, choices=[1, 2, 3], default=2, help='Case for data split: 1, 2, or 3')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Ratio for training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio for validation data')
    parser.add_argument('--train_data', type=str, nargs='+', help ='List of data keys to train the model on')
    parser.add_argument('--transform_data', type=str, nargs='+', help ='List of data keys to transform after running the model')

    
    
    args = parser.parse_args()
    args.name ="buddy"  # Imposta un nuovo nome del soggetto
    args.model_type="cebra_behavior"
    args.use_grid=False
    args.train_data=['X']
    print(f"processing data of {args.name}")
    
    #à# dynamic name
    #results_dict_name = f"{args.name}_results_dict"
    results = run_pipeline(
        args.input_dir,
        args.output_dir,
        args.name,
        args.param_file,
        args.model_type,
        args.use_grid,
        args.case,
        args.train_ratio,
        args.val_ratio, 
        args.train_data,
        args.transform_data
        
    )








