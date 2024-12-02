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
#main_root=r'F:\CNR_neuroscience\Consistency_across\Codice Davide\EEG-ANN-Pipeline'

## directories su ubuntu
import os
import sys
from pathlib import Path

# Add the parent directory to the system path

#input_dir=r'F:\CNR_neuroscience\Consistency_across\Codice Davide'
#main_root=r'F:\CNR_neuroscience\Consistency_across\Codice Davide\EEG-ANN-Pipeline'
#sys.path.append(main_root)
sys.path
#os.chdir(input_dir)


main_root = r"/media/zlollo/UBUNTU 24_0/CNR_neuroscience/Consistency_across/Codice Davide/EEG-ANN-Pipeline/"

input_dir=r"/media/zlollo/UBUNTU 24_0/CNR_neuroscience/Consistency_across/Codice Davide/"

output_dir=input_dir
os.chdir(input_dir)
if main_root not in sys.path:
    sys.path.append(main_root)
import json
import copy
import time
import yaml
import hashlib
import pickle
import warnings
import logging
import random
import typing
import argparse
from io import BytesIO
import pandas as pd
import torch
from torch import nn
import scipy.sparse as sp
from scipy import stats
import sympy
import h5py
import multiprocessing
from numpy.lib.stride_tricks import as_strided
os.getcwd()
from pathlib import Path
import joblib as jl
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('TkAgg')  
from torch.utils.data import DataLoader
from data.preprocessing import normalize_signals
#from data.eeg_dataset import *
from data import TrialEEG, DatasetEEG, DatasetEEGTorch
from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
from models import EncoderContrastiveWeights
from data import LabelsDistance
#plt.ion()  
#plt.show()
#plt.pause(10)  # Mostra il grafico per 10 secondi
#input("Premi Invio per continuare...")
from matplotlib.collections import LineCollection
from matplotlib.markers import MarkerStyle
import seaborn as sns
from joblib import Parallel, delayed

#from concurrent.futures import ProcessPoolExecutor
#from multiprocessing import Pool

from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import umap

import cebra.datasets
from cebra import CEBRA
from cebra.datasets.hippocampus import *

 # Sostituisci con una funzione specifica presente in some_functions.py
from some_functions import *
from model_utils import *

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

def run_pipeline(input_dir, output_dir, name, param_file, model_type, use_grid, case, train_ratio, val_ratio):
    #name='achilles'
    path = Path(input_dir) / f"{name}.jl"
    data = jl.load(path)
    data['X']=data['spikes']
    data['y']=data['position']
    
    
    #chns=data['spikes'].shape[1]
        #print(data['spikes'].shape)
        #print(data['position'].shape)
    #print(chns)
    
    fs=40
    data['trials'], c_t = create_trial_ids(data['position'][:,1])
    data['time_sec']=np.arange(0,len(data['spikes']))/fs
    
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
    params_path = Path(input_dir) / param_file
    params = load_params(params_path)
    fixed_params = params[model_type]['fixed']
    grid_params = params[model_type].get('grid', {})
    #batch_size = fixed_params.get('batch_size', 32)

    # paramete list
    if use_grid:
       param_list = list(ParameterGrid(grid_params))
    else:
       param_list = [{**fixed_params, **{k: v[0] for k, v in grid_params.items()}}]
    # Step 6: Processa i dati e ottieni le embedding
    

    

    
    embedding_results, data_split, temp, max_iters=process_data(data_split, param_list, fixed_params, model_type)
    
    z_train=embedding_results['embeddings_train']
    
    z_val=embedding_results['embeddings_valid']
    
    
    labels_train=data_split['y_train']
    labels_val=data_split['y_val']
    
    posdir_decode_CEBRA = decoding_knn(z_train, z_val, labels_train, labels_val)
    #temperature = fixed_params.get('temperature', 'N/A')
    #max_iters = grid_params.get('max_iterations', 'N/A')
    title = f"{model_type} - {name} Train Embeddings | Temp={temp} | Iterations={max_iters}"
    plot_embs(z_train, labels_train,title)
    #plt.show() 
    #posdir_decode_CL = decoding_knn(z_train, z_val, labels_train, labels_val)
    #plot_embs(embedding_results['embedding_train'],data_split['y_train'])
    print('test_score is', posdir_decode_CEBRA[0])
    print('pos test error is', posdir_decode_CEBRA[1])
    print('pos test score is', posdir_decode_CEBRA[2])
    return embedding_results , data_split, posdir_decode_CEBRA
    ####################### Function to laod parameteres #########################
    
  
    
 
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizzo del dispositivo: {device}")
    input_dir = os.getcwd()
    output_dir = input_dir

    parser = argparse.ArgumentParser(description='Esegui la pipeline di elaborazione dati e embedding.')
    parser.add_argument('--input_dir', type=str, default=input_dir, help='Input directory path')
    parser.add_argument('--output_dir', type=str, default=output_dir, help='Output directory path')
    parser.add_argument('--param_file', type=str, default='model_params_1.yaml', help='Parameter file in YAML format')
    parser.add_argument('--model_type', type=str, default='cebra_behavior', help='Type of model to use')
    parser.add_argument('--name', type=str, default='achilles', help='Name of the subject (file name without extension)')
    parser.add_argument('--use_grid', type=bool, default=False, help='Whether to use grid search for parameters')
    parser.add_argument('--case', type=int, choices=[1, 2, 3], default=2, help='Case for data split: 1, 2, or 3')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Ratio for training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio for validation data')
    
    
    
    args = parser.parse_args()
    args.name = "achilles"  # Imposta un nuovo nome del soggetto
    print(f"processing data of {args.name}")
    embedding_results, data_split, posdir_decode_CEBRA=run_pipeline(args.input_dir, args.output_dir, args.name, args.param_file, args.model_type, args.use_grid, args.case, args.train_ratio, args.val_ratio)















