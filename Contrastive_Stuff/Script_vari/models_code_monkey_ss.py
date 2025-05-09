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
### directory su windows
#input_dir=r'F:\CNR_neuroscience\Consistency_across\Codice Davide'

## directories su ubuntu
i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

####sample notebook.
'''

'''
import os
import sys
from pathlib import Path

os.chdir(i_dir)
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
from cebra.datasets.hippocampus import *
 # Sostituisci con una funzione specifica presente in some_functions.py
from some_functions import *
from model_utils import *

#input("Premi Invio per continuare...")

#from concurrent.futures import ProcessPoolExecutor
#from multiprocessing import Pool
#plt.ion()



def process_data(data_split, param_list, fixed_params, model_type):
    """
    Processa i dati e esegue il modello utilizzando la funzione run_model
    """
    
    #embedding_results = {} 
    
    for p in param_list:
        print(f"\nEsecuzione del modello {model_type} con parametri: {p}")
        model_params = {**fixed_params, **p}

        # Esegui `run_model` con i dati di input e i parametri specifici
        results = run_model(
            model_type=model_type,
            params=model_params,
            data_=data_split,
            subtrain=('_subtrain' in data_split)
        )

        if results[0] is None:
            continue

        embeddings_train, embeddings_valid, embeddings_sub_train, fitted_model, y_train, y_valid, y_sub_train, X_sub_train,train_loss = results
        # Uti
        
        key = f"{model_type}_{str(p)}"
        
        embedding_results={
            'model_type':model_type,
             'model_params':model_params,
             'embeddings_train':embeddings_train,
             'embeddings_valid':embeddings_valid,
             'embeddings_subtrain':embeddings_sub_train,
             'train_loss': train_loss}
        
        temp= model_params.get('temperature', 'N/A')
        max_iters = model_params.get('max_iterations', 'N/A')

        #if isinstance(train_loss, torch.Tensor):
       #     last_train_loss = train_loss.item()
       # elif isinstance(train_loss, list) and isinstance(train_loss[-1], torch.Tensor):
       #     last_train_loss = train_loss[-1].item()
       # else:
       #     last_train_loss = train_loss
    
        #input("Premi Invio per chiudere il grafico degli embedding e continuare...")
  #sappend((model_type, model_params,embeddings_train, embeddings_valid, embeddings_sub_train))

        
    return embedding_results, data_split, temp, max_iters

def run_pipeline(input_dir, output_dir, param_file,label_type, model_type, use_grid, case, train_ratio, val_ratio):
    #name='achilles'
    path = Path(input_dir) / "macaque_data.jl"
    data = jl.load(path)
    data['X']=data['spikes_active']
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
    
    #chns=data['spikes'].shape[1]
        #print(data['spikes'].shape)
        #print(data['position'].shape)
    #print(chns)
    
    
    #data['trials'], c_t = create_trial_ids(data['position'][:,1])
    #data['time_sec']=np.arange(0,len(data['spikes_active']))/fs
    
    #tr_r=0.85
    #val_r=1-t_r
    #test_r=0
    
    ### SPLIT DATA
    # Case 1: Train, Validation, and Test
    # Case 2: Train and Validation only
    # Case 3: Train (Subtrain and Validation within Train) and Test
    
    ### keep case 2
    
    #data_split = split_data_trials(data, case=case, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=False, seed=42, verbose=True)    
    data_split = split_data_trials(data, case=case, train_ratio=train_ratio, val_ratio=val_ratio, shuffle=False, seed=42, verbose=True)
    print("Chiavi in data_split:", data_split.keys())
    params_path = Path(project_root) / param_file
    params = load_params(params_path)
    fixed_params = params[model_type]['fixed']
    grid_params = params[model_type].get('grid', {})
    #batch_size = fixed_params.get('batch_size', 32)

    # paramete list
    if use_grid:
       param_list = list(ParameterGrid(grid_params))
    else:
       param_list = [{**fixed_params, **{k: v[0] for k, v in grid_params.items()}}]
   
    
   # Model processing to getn embeddings

    
    embedding_results, data_split, temp, max_iters=process_data(data_split, param_list, fixed_params, model_type)
    
    z_train=embedding_results['embeddings_train']
    
    z_val=embedding_results['embeddings_valid']
    
    
    labels_train=data_split['y_train']
    labels_val=data_split['y_val']
    
    #posdir_decode_CEBRA = decoding_knn(z_train, z_val, labels_train, labels_val)
    #temperature = fixed_params.get('temperature', 'N/A')
    #max_iters = grid_params.get('max_iterations', 'N/A')
    title = f"{model_type} trained with {label_type} label | Temp={temp} | Iterations={max_iters}"

    plt_emb(z_train, labels_train,title)
    #plt.show() 
    #posdir_decode_CL = decoding_knn(z_train, z_val, labels_train, labels_val)
    #plot_embs(embedding_results['embedding_train'],data_split['y_train'])
    #print('test_score is', posdir_decode_CEBRA[0])
    #print('pos test error is', posdir_decode_CEBRA[1])
    #print('pos test score is', posdir_decode_CEBRA[2])
    return embedding_results , data_split
#, posdir_decode_CEBRA
    ####################### Function to laod parameteres #########################
    
  
 
if __name__ == "__main__":
    project_root, default_output_dir, default_input_dir = setup_paths()

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    input_dir = default_input_dir
    output_dir = default_output_dir
    
    

    parser = argparse.ArgumentParser(description='Esegui la pipeline di elaborazione dati e embedding.')
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Output directory path')
    parser.add_argument('--param_file', type=str, default='model_params_1.yaml', help='Parameter file in YAML format')
    parser.add_argument('--model_type', type=str, default='cebra_behavior', help='Type of model to use')
    parser.add_argument('--label_type', type=str, default='continuous', help='type of chosen label')
    parser.add_argument('--use_grid', type=bool, default=False, help='Whether to use grid search for parameters')
    parser.add_argument('--case', type=int, choices=[1, 2, 3], default=2, help='Case for data split: 1, 2, or 3')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Ratio for training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio for validation data')
    
    
    
    args = parser.parse_args()
    args.label_type = "discrete"  # Imposta un nuovo nome del soggetto
    print(f"processing data of {args.label_type}")
    
    # , posdir_decode_CEBRA 
    embedding_results, data_split= run_pipeline(
        args.input_dir,
        args.output_dir,
        args.param_file,
        args.label_type,
        
        args.model_type,
        args.use_grid,
        args.case,
        args.train_ratio,
        args.val_ratio
    )













