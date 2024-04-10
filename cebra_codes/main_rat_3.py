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
os.chdir(main_path)

os.getcwd()

params = {
    "model_architecture": 'offset10-model',
    "batch_size": "512",
    "learning_rate": "3e-4",
    "temperature": "1",
    "output_dimension": "3",
    "max_iterations": "10000",
    "distance": "cosine",
    "conditional": "time_delta",
    "hybrid": "True",
    "time_offsets": "10",
    "seed": "32"
    }



from hip_models_1 import run_hip_models
from fig_cebra_1 import plot_cebra
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_manif
import random

def main(params):
    #create_database() 
    base_path=main_path
      
    random.seed(1);
    maxI = 10^5;
    N = 100
    group_name= 'Cebra_behav'
    ### create a group of manifold per label
    include_labels = True  # vel False, sup vs unsup
           
    for n in range(1, N+1):
        params['seed']=random.randint(1,maxI)
        manif, labels = run_hip_models(base_path,params)
        file_name = "manif_data.hdf5"

        labels = labels  # o le tue labels, se hai deciso di includerle

        with create_or_open_hdf5(file_name) as hdf5_file:
            save_manif(hdf5_file, group_name, manif, labels=labels,
            include_labels=include_labels)
        
        # Opional: plotting data
        # fig=plot_cebra(manif, labels)
        # save_fig(fig, fig_id="my_plot_id")
        #plot_cebra(manif, labels)

        #input("Press any key to continue..")

    return manif, labels
    
if __name__=="__main__":

    main(params)


file_name = 'manif_data.hdf5'
try:
    with h5py.File(file_name, 'r') as f:
        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
except Exception as e:
    print(e)
