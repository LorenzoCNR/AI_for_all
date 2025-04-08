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
i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

#  ubuntu directories
#i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

os.chdir(i_dir)
os.getcwd()
from some_functions import *
import json
import copy
import time
import numpy as np
import yaml
import pickle
import warnings
import logging
#import umap 
import openTSNE
import random
import typing
import joblib as jl
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
from scipy import optimize as opt

from cebra.datasets.hippocampus import *
 # Sostituisci con una funzione specifica presente in some_functions.py
from model_utils import *
# Random Seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True



# 2) NAME of folder containing data (input) directory. 
data_dir="data"
# (specific) project data folder
sub_data="Monkeys_Mirco"

# 3) PIPELINE folder name
pipe_path= "EEG-ANN-Pipeline"

# 4) OUTPUT folder: folder to store processed output
#    (if not existing is created)
out_dir="contrastive_output"

project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths( data_dir,sub_data,out_dir, pipe_path,change_dir=False)

# recall the input dir (declared upwards) and import data
input_dir = default_input_dir
# data format
d_format="mat"
# data_name
d_name='dati_mirco_18_03_joint'
data = load_data(input_dir, d_name,d_format)
print(type(data)) # must be a dictionary

     
#X=data['m1_active_neural'][:,[0,2]]
X=data['joint_mix_neural']
y_dir=data['joint_mix_trial']
y_dir=y_dir.flatten()
#y_pos=data['mix_active_trial']
trial_id=data['joint_mix_trial_id']
trial_id=trial_id.flatten()
len_y=len(y_dir)
change_idx = np.where(np.diff(trial_id) != 0)[0] + 1
change_idx
print('ciao')
# the c_t vector tells the starting and endiing points of every trial
c_t=np.concatenate([[0], change_idx,[len_y]], dtype=int)
## list of list of starting and ending points for trials

c_t_list=[]
c_t_list = [(c_t[i], c_t[i+1] - 1) for i in range(len(c_t) - 1)]
n_trials=len(c_t)-1
### check trial length (useful for graphics)
trial_len=np.diff(c_t)
trial_length=trial_len[0]
original_label_order = np.arange(1, 9)  # [1,2,3,4,5,6,7,8]

# direction_dict = {
#     1: 'nord',
#     2: 'nord_est',
#     3: 'est',
#     4: 'sud_est',
#     5: 'sud',
#     6: 'sud_ovest',
#     7: 'ovest',
#     8: 'nord_ovest'
# }

########################### RESAMPLING ########################################
#### freqeuncy of sampling(if you want to use data at a lower freq)
#sampling_freq=10

##### option to resample data a different frequency 
#### RESAMPLIGN DATA FUNCTION
for start_trial, end_trial in c_t_list:
    print(start_trial, end_trial)


methods = {
    0: "mean",  
    1: "center"  
    #,2: "mean"
}

l_data=[X,y_dir] 
step=10
overlap=5
Normalize=True
### resampled data, new trials lengths, new trials intervals
resampled,r_trial_lengths,r_trial_indices =  f_resample(l_data,c_t_list, step,
                       overlap, methods, mode="overlapping",normalization=True)

r_trial=r_trial_indices[0]
start_points=[]
for sublist in r_trial:
    print(sublist[0])
    start_points.append(sublist[0])
r_last=r_trial[-1][1]
start_points.append(r_last+1)
c_t_resampled=np.array(start_points).flatten()

r_trial_indices[0][1][0]
unique_labels = np.unique(resampled[1])
### 6-3, 3-6

swap_dict = {3: 6, 6: 3}

# Apply mapping back to restore original labels (optional)
resampled_swapped = swap_labels(resampled[1], swap_dict)



### ricampionamento e permutazione delle labels ###

X_=X
y_dir_=y_dir
data_={"X":X_,"y":y_dir_}
train_data_=['X', 'y']
transform_data_=['X']

### run and validate model (store loss del train...facile) ma se voglio

c_s="maroon"
output_folder = default_output_dir
results_list=[]
#title='CEBRA-behavior trained with target label'
ww=0
trial_length=trial_len[0]
y_dir_=y_dir_.flatten()
#☻plot_embs_discrete(X_hat,y_dir_, title,trial_length, ww)

constant_len=True



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
     title_=f"CEBRA_cond3_shift 5_mean_lr_{l_r}_nhu_{n_h_u}_temp{temp}.html"

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
     
     plot_direction_averaged_embedding(
            X_hat,
            y_dir_,
            original_label_order,
            c_s,
            output_folder,
            title_,
            trial_length,
            constant_length=const_len,
            ww=0,
            label_swap_info=None
        ) 

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
     
     
     # if not results:
     #     continue
            
 # Model processing to getn embeddings


# second
#n_traj=np.arange(8)+9
#
fig = plt.figure(figsize=(4, 2), dpi=300)
plt.suptitle('CEBRA-behavior trained with target label',
             fontsize=5)
ax = plt.subplot(121, projection = '3d')
ax.set_title('All trials embedding', fontsize=5, y=-0.1)
x = ax.scatter(X_hat[:, 0],
               X_hat[:, 1],
               X_hat[:, 2],
               c=y_dir_,
               cmap=plt.cm.hsv,
               s=0.01)
ax.axis('off')

ax = plt.subplot(122,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
for i in range(n_traj):
    direction_trial = (y_dir_ == i+1)
    selected = X_hat[direction_trial, :]
    
    if selected.shape[0] == 0:
        print(f"Attenzione: nessun trial per la direzione {i}")
        continue  # oppure salta il calcolo, oppure assegna np.nan
    trial_avg =X_hat[direction_trial, :].reshape(-1, trial_length-ww,
                                                         3).mean(axis=0)
    trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    ax.scatter(trial_avg_normed[:, 0],
               trial_avg_normed[:, 1],
               trial_avg_normed[:, 2],
               color=plt.cm.hsv(1 / 8 * i),
               s=0.01)
ax.axis('off')
plt.show()


#from some_functions import *







# ### either continuous for position or discrete for target 
# label_type='discrete'
# if label_type=="discrete":
#     data['y']=data['active_target']
#     from  some_functions import plot_embs_discrete as plt_emb

# elif label_type == "continuous":
#         data['y'] = data['pos_active']
#         from  some_functions import plot_embs_continuous as plt_emb

# else:
#      data['y'] = data['pos_active']

 
# fs=1000
# data['trials']=np.repeat(np.arange(1,194, ), 600)
# data['time']=np.arange(0,115800,)/1000
######################## Pre process of data #############################
### to do (what and how?)



 #################################### MODEL PART ##############################


'''
'''
# Split parameters
train_ratio = 0.85
val_ratio_cv = 0.15
num_splits = 3  # Numero di split per CV
seed = 42

# Initail split
train_trials, val_trials = circular_initial_split(trial_id, train_ratio, seed=seed)

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

model_type='cebra_behavior'

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
        subtrain_idx = np.isin(trial_id, subtrain_trials).nonzero()[0]
        cv_val_idx = np.isin(trial_id, cv_val_trials).nonzero()[0]
        
        # Dati subtrain e validation per CV
        subtrain_X, subtrain_y = X[subtrain_idx], y_dir[subtrain_idx]
        cv_val_X, cv_val_y = X[cv_val_idx], y_dir[cv_val_idx]
        
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
        
        # # validation loss  (for behavior model add cv_val_y...i.e.label used to train the model)
        # val_loss = cebra.sklearn.metrics.infonce_loss(
        #     model['fitted_model'], cv_val_X, num_batches=100, correct_by_batchsize=False
        # )
        # print(f"Split {split_idx}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")
        # cumulative_val_loss += val_loss

    # Media delle loss
    avg_train_loss = cumulative_train_loss / num_splits
    # avg_val_loss = cumulative_val_loss / num_splits
    
    # Salva risultati
    results = {
        "model":model['fitted_model'],
        "params": p,
        "avg_train_loss": avg_train_loss,
       # "avg_val_loss": avg_val_loss,
    }
    results_list.append(results)

# BEst model and parameters
best_result_continuous = min(results_list, key=lambda x: x['avg_train_loss'])
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


'''
