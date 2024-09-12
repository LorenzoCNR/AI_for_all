# -*- coding: utf-8 -*-
"""
@author: zlollo
"""
import os
#os.chdir(r'/media/zlollo/STRILA/CNR neuroscience/cebra_codes')
#input_directory=os.getcwd()
#output_directory=input_directory

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
import joblib as jl
from scipy.io import loadmat, savemat
import sys

def load_data(input_directory, data_filename='rat_n.mat'):
    
    data_path = os.path.join(input_directory, data_filename)
    data_mat = loadmat(data_path)
    return data_mat['rat_n'] 

#m_params    = loadmat(os.path.join(input_directory,'umap_params.mat'))
#model_params        = m_params['params']
#print(model_params)


def load_params(input_directory,params_filename ):
    try:
        m_params = loadmat(params_filename)
        return m_params['params'][0, 0]
    except FileNotFoundError:
        print(f"file not found!!!.")
        return None

### check how parameters are loaded:
    # guideline
    # string... metric=params['metric'][0].item()
    # literals... r_s=params['random_state'][0].item()==None
    #             a_r_f=params['angular_rp_forest'][0].item()==False

#a_r_f = {'True': True, 'False': False, 'None': None}.get(params['angular_rp_forest'][0], params['angular_rp_forest'][0])


def configure_and_fit_umap(data, params):
    umap_model = umap.UMAP(
        n_neighbors=int(params['n_neighbors'][0][0]),
        n_components=int(params['n_components'][0][0]),
        min_dist=float(params['min_dist'][0][0]),
        #learning_rate='auto' if params['learning_rate'] == 'auto' else float(params['learning_rate'][0, 0]),
        random_state={'True': True, 'False': False, 'None': None}.get(params['angular_rp_forest'][0], params['angular_rp_forest'][0]),
        metric=params['metric'][0].item()
    )
    return umap_model.fit(data)

def save_embedding(embedding, output_directory, filename='umap_embedding.mat'):
    
    embedding_path = os.path.join(output_directory, filename)
    savemat(embedding_path, {'embedding': embedding})
    #print(f"Embedding saved to {embedding_path}")

def save_model(model, output_directory, model_name='umap_model.pkl'):
    model_path = os.path.join(output_directory, model_name)
    jl.dump(model, model_path)
    #print(f"Model saved to {model_path}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <input_directory> <output_directory> <data_filename> <params_filename>")
        sys.exit(1)

    input_directory, output_directory, data_filename, params_filename = sys.argv[1:5]
    
    # ## test
    #input_directory 
    #output_directory 
    #data_filename = 'rat_n.mat'
    #params_filename = 'umap_params.mat'
    # load data and params
    data_ = load_data(input_directory, data_filename)
    params_ = load_params(input_directory, params_filename)

    ## configure and train model


    umap_model = configure_and_fit_umap(data_, params_)
    

    # Save manifold or embedding :)
    #save_embedding(tsne_model.embedding_, output_directory, 'tsne_embedding.mat')

    # Save model
    save_model(umap_model, output_directory, 'umap_model.pkl')
    #umap_model.transform(data)
    
    