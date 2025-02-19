# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:45:45 2024

@author: loren
"""
'''
Il codice prende i dati di un soggetto (ratto) divide in trial, split in train test validation etc
processa i dati secondo il modello scelto 
 USO
Da terminale lanciare cmabiando gli args del parser all'occorrenza
python script_name.py --input_dir "path/to/input" --output_dir "path/to/output" --param_file "model_params.yaml" --model_type "cebra_behavior" --name "achilles" --use_grid False --case 2 --train_ratio 0.85 --val_ratio 0.15


DA FARE
mettere nel file params o in parser, la metrica per il decoding e il numero di vicini
raffianre il decoding con scelta di k ottimale



'''


############################### METTERE LE PROPRIE DIRECTORIES!!!!!! #######################################
##
####sample notebook.
'''

'''

import os
import sys
from pathlib import Path
### directory su windows
i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

## directories su ubuntu
#i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

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
os.chdir(i_dir)


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
   
    default_input_dir= project_root.parent/ "data" / "monkey_reaching_preload_smth_40"
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
path = Path(input_dir) / "macaque_data.jl"
data = jl.load(path)
data['X']=data['spikes_active']
### either continuous for position or discrete for target 
label_type='discrete'
if label_type=="discrete":
    data['y']=data['active_target']
    from  some_functions import plot_embs_discrete as plt_emb

elif label_type == "continuous":
        data['y'] = data['pos_active']
        from  some_functions import plot_embs_continuous as plt_emb

else:
     data['y'] = data['pos_active']
     

 
fs=1000
data['trials']=np.repeat(np.arange(1,194, ), 600)
data['time']=np.arange(0,115800,)/1000
######################## Pre process of data #############################
### to do (what and how?)



 #################################### MODEL PART ##############################
from some_functions import *
from model_utils import *

# Split parameters
train_ratio = 0.85
val_ratio_cv = 0.15
num_splits = 3  # Numero di split per CV
seed = 42

# Initail split
train_trials, val_trials = circular_initial_split(data['trials'], train_ratio, seed=seed)

# Risultati
results_list = []

### point to the (hyper)parameters file
### params file is a yaml

#from cebra.sklearn.metrics import infonce_loss
param_file='model_params_1.yaml'
params_path = Path(project_root) / param_file
params = load_params(params_path)

## define the model and declare the data u want to use
### model type: cebra_time, cebra_behavior, cebra_hybrid, UMAP, TSNE, covn_pivae

model_type='cebra_time'

fixed_params = params[model_type]['fixed']
grid_params = params[model_type].get('grid', {})
#param_list = list(ParameterGrid(grid_params))  # o ParameterSampler per random search

### random search on parameter to limit the combo
n_iter=20
random_search=ParameterSampler(grid_params, n_iter=n_iter, random_state=42)
param_list = list(random_search)
print(param_list)
print(grid_params)
# Risultati
results_list = []

for p in param_list:
    print(f"\nTesting parameters: {p}")
    model_params = {**fixed_params, **p}
    
    cumulative_val_loss = 0
    cumulative_train_loss = 0

    for split_idx in range(num_splits):
        # Split circolare sui trial
        subtrain_trials, cv_val_trials = circular_cv_split(train_trials, train_trials, val_ratio_cv, split_idx)

        # Indici per subtrain e val
        subtrain_idx = np.isin(data['trials'], subtrain_trials).nonzero()[0]
        cv_val_idx = np.isin(data['trials'], cv_val_trials).nonzero()[0]
        
        # Dati subtrain e validation per CV
        subtrain_X, subtrain_y = data['X'][subtrain_idx], data['y'][subtrain_idx]
        cv_val_X, cv_val_y = data['X'][cv_val_idx], data['y'][cv_val_idx]
        
        # Training
        model = run_model(
            model_type=model_type,
            params=model_params,
            data_={"X_train": subtrain_X, "y_train": subtrain_y},
            train_data=["X_train", "y_train"],
            save_results=False,
        )
        
        # training loss
        train_loss = model['fitted_model'].state_dict_['loss'][-1].numpy()
        cumulative_train_loss += train_loss
        
        # validation loss  (for behavior model add cv_val_y...i.e.label used to train the model)
        val_loss = cebra.sklearn.metrics.infonce_loss(
            model['fitted_model'], cv_val_X, num_batches=100, correct_by_batchsize=False
        )
        print(f"Split {split_idx}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        cumulative_val_loss += val_loss

    # Media delle loss
    avg_train_loss = cumulative_train_loss / num_splits
    avg_val_loss = cumulative_val_loss / num_splits
    
    # Salva risultati
    results = {
        "model":model['fitted_model'],
        "params": p,
        "avg_train_loss": avg_train_loss,
        "avg_val_loss": avg_val_loss,
    }
    results_list.append(results)

# BEst model and parameters
#best_result_continuous = min(results_list, key=lambda x: x['avg_val_loss'])
print("\nBest Result:", best_result_continuous)
#embed_continuous=best_result['model'].transform(data['X'])
#label_cont=data['y']
label_cont=data['pos_active']
title='CEBRA-behavior trained with position label'
plot_embs_continuous(embed_continuous, label_cont, title)
print(best_result_continuous['params'])
#{'temperature': 1, 'num_hidden_units': 64, 'max_iterations': 4000, 'learning_rate': 0.0001}


#best_result_discrete = min(results_list, key=lambda x: x['avg_val_loss'])
print("\nBest Result:", best_result)
#embed_discrete=best_result['model'].transform(data['X'])
label_disc=data['y']
#label_disc=data['active_target']
title='CEBRA-behavior trained with target label'
plot_embs_discrete(embed_discrete, label_disc, title)
print(best_result_discrete['params'])
#{'temperature': 1, 'num_hidden_units': 64, 'max_iterations': 4000, 'learning_rate': 0.0001}


best_result_time = min(results_list, key=lambda x: x['avg_val_loss'])
print("\nBest Result:", best_result)
embed_time=best_result['model'].transform(data['X'])
label_time=data['pos_active']
title='CEBRA-behavior trained with target label'
plot_embs_continuous(embed_time, label_time, title)
print(best_result_time['params'])
#{'temperature': 1, 'num_hidden_units': 32, 'max_iterations': 6000, 'learning_rate': 0.0028521675677231
#### once the model is found
#### either porceed with decoding or gnerate embedding and plos




######################## Pre process of data #############################
### to do (what and how?)

################# Data Split
### split mode 
# case 1 Train, Validation, and Test
# case 2 Train and Validation only 
# case 3 Train (Subtrain and Validation within Train) and Test
case=2
train_ratio=1
### non occorre in case 2 (solo train e valid)
val_ratio=0

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
     ## SE STIMI SENZA VALIDARE COMMENTA LE RIGHE SUCCESSIVE 
     ### compute infonce also on validation data: comment to skip it 
     ### refinement: provide the option with data
     # results['valid loss'] = cebra.sklearn.metrics.infonce_loss(
     #     results['fitted_model'],
     #     validation_X,
     #     validation_y,
     #     num_batches=100,  # Numero di batch da considerare per la valutazione
     #     correct_by_batchsize=False,  # Correzione basata sulla batch size
     # )
     
    #results_list.append(results)
     
     #min_val_loss=min(results_list, key=lambda d: d['valid loss'])
     #min_train_loss=min(results_list, key=lambda d: d['train final loss'])
     
     
     if not results:
         continue
            
 # Model processing to getn embeddings
from some_functions import *
from model_utils import *
embedding_results=results['fitted_model'].transform(data_split['X'])
title='CEBRA-behavior trained with target label'
plot_embs_discrete(embedding_results,data_split['y'] , title)