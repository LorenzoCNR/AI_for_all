#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:47:54 2023

@author: zlollo
"""

import os

import matplotlib
import h5py
matplotlib.use('TkAgg')


#main_path=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'
main_path=r'/home/donnarumma/tools/Cebra_for_all/cebra_codes'
#os.chdir(main_path)

#os.getcwd()

params = {
    "model_architecture": 'offset10-model',
    "batch_size": "512",
    "learning_rate": "3e-4",
    "temperature": "1",
    "output_dimension": "3",
    "max_iterations": "10000",
    "distance": "cosine",
    "conditional": "time_delta",
    "hybrid": "False",
    "time_offsets": "10",
    "seed": "35"
    }

from hip_models_1 import run_hip_models
from fig_cebra_1 import plot_cebra
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_manif
import random

def main(params):
    #create_database() 
    base_path=main_path
    
    random.seed(params['seed']);
    maxI = 10^5;
    N = 1
    group_name= 'data'
    ### create a group of manifold per label
           
    for n in range(1, N+1):
        params['seed']=random.randint(1,maxI)
        manif, labels = run_hip_models(base_path,params)
        output_filename = "encoded_manifold.hdf5"
        input('Wait')
        # save data in 
        with h5py.File(output_filename, 'w') as out_hdf:
            group = out_hdf.create_group(group_name)
            group.create_dataset("manifold", data=manif,  compression='gzip', compression_opts=9)
            group.create_dataset("labels",   data=labels, compression='gzip', compression_opts=9)
            print(f"Transformed data saved in {output_filename}")
        
    return manif, labels
    
if __name__=="__main__":

    main(params)

    output_filename = "encoded_manifold.hdf5"
        
try:
    with h5py.File(output_filename, 'r') as f:
        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
except Exception as e:
    print(e)
