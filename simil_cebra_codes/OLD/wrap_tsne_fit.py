# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 00:39:20 2024

@author: loren
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
from scipy.io import loadmat, savemat
import openTSNE



###from sklearn.model_selection import train_test_split



### N.B. Ricorda di chiamare in matlab il file parametri params
### deve essere l'input del file compute
#m_params    = loadmat(os.path.join(input_directory,'tsne_params.mat'))
#model_params        = m_params['params']
#print(model_params)

# try:
#     # Assumi che tutti i parametri siano presenti nell'array strutturato.
#     # Estrai ciascun parametro verificando il suo tipo e convertendolo di conseguenza.
#     initialization = model_params['initialization'][0].item()
#     initialization = initialization[0]
    
    
#     perpl = float(model_params['perplexity'][0][0])

#     theta = float(model_params['theta'][0][0])

#     dof = int(model_params['dof'][0][0])

#     n_j = int(model_params['n_jobs'][0][0])
    
    
#     metric = model_params['metric'][0].item()
#     metric = metric[0]
  
#     # both string and float (default ='auto')
#     #learn_rate = float(model_params['learning_rate'][0][0]) 
    
#     iters = model_params['n_iter'][0].item()
#     iters = iters[0]

#     n_comp = int(model_params['n_components'][0][0])

#     verb = model_params['verbose'][0].item() == 'True'
  
# except FileNotFoundError:
#     print(f"model_params file is missing!!!")

def load_data(input_directory, data_filename='rat_n.mat'):
    
    data_path = os.path.join(input_directory, data_filename)
    data_mat = loadmat(data_path)
    return data_mat['rat_n'] 


def load_params(input_directory,params_filename ):
    try:
        m_params = loadmat(params_filename)
        return m_params['params'][0, 0]
    except FileNotFoundError:
        print(f"file not found!!!.")
        return None

def configure_and_fit_tsne(data, params):
    tsne = openTSNE.TSNE(
        n_components=int(params['n_components'][0][0]),
        perplexity=float(params['perplexity'][0][0]),
        learning_rate='auto' if params['learning_rate'] == 'auto' else float(params['learning_rate'][0, 0]),
        n_iter=int(params['n_iter'][0][ 0]),
        theta=float(params['theta'][0][0]),
        verbose=params['verbose'][0].item() == 'True'
    )
    return tsne.fit(data)

def save_embedding(embedding, output_directory, filename='tsne_embedding.mat'):
    
    embedding_path = os.path.join(output_directory, filename)
    savemat(embedding_path, {'embedding': embedding})
    #print(f"Embedding saved to {embedding_path}")

def save_model(model, output_directory, model_name='tsne_model.pkl'):
    model_path = os.path.join(output_directory, model_name)
    jl.dump(model, model_path)
    #print(f"Model saved to {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_directory> <output_directory> <data_filename> <params_filename>")
        sys.exit(1)

    input_directory, output_directory, data_filename, params_filename = sys.argv[1:5]
    
    # ## test
    # input_directory 
    # output_directory 
    # data_filename = 'rat_n.mat'
    # params_filename = 'tsne_params.mat'
    # load data and params
    data_ = load_data(input_directory, data_filename)
    params_ = load_params(input_directory, params_filename)

    ## configure and train model

    tsne_model = configure_and_fit_tsne(data_, params_)

    # Save manifold or embedding :)
    #save_embedding(tsne_model.embedding_, output_directory, 'tsne_embedding.mat')

    # Save model
    save_model(tsne_model, output_directory, 'tsne_model.pkl')