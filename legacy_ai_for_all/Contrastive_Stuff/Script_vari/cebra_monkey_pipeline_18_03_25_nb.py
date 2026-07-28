# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:45:45 2024

@author: loren
"""
'''


'''

### Just remember to provide the data (Mirco) with description
### Describe Methods (of resampling)
## Add an option to plot offline (off browser to say better)
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
import json
import copy
import time
#import shap
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
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split,  ParameterSampler, RandomizedSearchCV
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

# DECLARE 
# 1) PROJECT root directory
#  windows directories
i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

#  ubuntu directories
#i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

os.chdir(i_dir)
os.getcwd()
from some_functions import *
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
d_name='dati_mirco_18_03_k'
data = load_data(input_dir, d_name,d_format)
print(type(data)) # must be a dictionary

     
#X=data['m1_active_neural'][:,[0,2]]
X=data['k_cond2_active_neural']
y_dir=data['k_cond2_active_trial']
y_dir=y_dir.flatten()
#y_pos=data['mix_active_trial']
trial_id=data['k_cond2_active_trial_id']
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

########################### RESAMPLING ########################################

##### option to resample data a different frequency 
#### RESAMPLIGN DATA FUNCTION
#for start_trial, end_trial in c_t_list:
    #print(start_trial, end_trial)

### choose resampling method according to variable type
methods = {
    0: "sum",  
    1: "center"  
    #,2: "mean"
}

## data to resample
l_data=[X,y_dir] 
step=10
overlap=6
Normalize=True
### resampled data, new trials lengths, new trials intervals
resampled,r_trial_lengths,r_trial_indices =  f_resample(l_data,c_t_list, step,
                       overlap, methods, mode="overlapping",normalization=True)

## new Trials' intervals
r_trial=r_trial_indices[0]
start_points=[]
start_points = [x[0] for x in r_trial] + [r_trial[-1][1] + 1]

c_t_resampled=np.array(start_points).flatten()

r_trial_indices[0][1][0]
unique_labels = np.unique(resampled[1])


### Swapping option
### ricampionamento e permutazione delle labels ###
### 6-3, 3-6
#swap_dict = {3: 6, 6: 3}
# Apply mapping back to restore original labels (optional)
#resampled_swapped = swap_labels(resampled[1], swap_dict)

################## when picking resampled  data ##########################
X_=resampled[0]
len_labs=len(X_)
y_dir_=resampled[1].reshape(len_labs,1).astype(int)
trial_length=r_trial_lengths[0][0]

################ when picking original data ###########################
#X_=X
#len_labs=len(X_)
#y_dir_=y_dir.reshape(len_labs,1).astype(int)
#♠trial_length=trial_len[0]



############## if we want to create a time variable to augment the label info
#y_time_=list(np.arange(1,trial_length+1))*n_trials
#y_time_=np.array(y_time_).reshape(len_labs,1).astype(float)

#yy_=np.concatenate((y_dir_,y_time_),axis=1)


################################### MODEL PART ##############################
################## model data
### neural data and direction labels
### with no resampling just pick X and y_dir

data_={"X":X_,"y":y_dir_}

train_data_=['X', 'y']
# neural data (full sample)
transform_data_=['X']
#from cebra.sklearn.metrics import infonce_loss
param_file='model_params_1.yaml'
params_path = Path(project_root) / param_file
params = load_params(params_path)

## define the model and declare the data u want to use
model_type='cebra_behavior'

fixed_params = params[model_type]['fixed']

model_params = {**fixed_params}
model_params

     #######
## you can change whatever parameter you want 
model_params['max_iterations']=20000
model_params['temperature']=1.266
model_params['num_hidden_units']=64

## set the model
cebra_label=CEBRA(**model_params)
# pippo=data_.get(train_data_[1])
## fit model
cebra_label.fit(data_.get(train_data_[0]),data_.get(train_data_[1]))

### transform data (reduced dimension data)
X_hat=cebra_label.transform(data_.get(transform_data_[0]))



### run and validate model (store loss del train...facile) ma se voglio


##################################### PLOT
c_s="maroon"
output_folder = default_output_dir
results_list=[]
#title='CEBRA-behavior trained with target label'
ww=0
y_dir_=y_dir_.flatten()
#☻plot_embs_discrete(X_hat,y_dir_, title,trial_length, ww)

const_len=True
### parameters' values to name the plot   
n_h_u=model_params['num_hidden_units']
temp=model_params['temperature']
iters=model_params['max_iterations']

### shift or steps is the number of overlapping points 
title_=f"K_CEBRA_cond1_shift_{overlap}_sum_nhu_{n_h_u}_temp{temp}_iters{iters}.html"
#results={}
#results={}
#results[title_]=X_hat

#results['train final loss']=cebra_label.state_dict_['loss'][-1].numpy()
#results['train loss']=cebra_label.state_dict_['loss'].numpy()

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
