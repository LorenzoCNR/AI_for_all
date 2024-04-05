#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:47:54 2023

@author: zlollo
"""

import os

import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

import joblib as jl
from matplotlib.collections import LineCollection
import inspect
import h5py
import torch
#import tensorflow as tf
from pathlib import Path
from datetime import datetime
##### cambiare eventualmente


#main_path=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'
main_path=r'/home/donnarumma/tools/Cebra_for_all/cebra_codes'
os.chdir(main_path)
#main_path=r'/home/zlollo/CNR/Cebra_for_all'

#os.chdir(main_path)

os.getcwd()
#from pathlib import Path


'''


'''
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
    "time_offsets": "10"
    }



from hip_models_1 import run_hip_models
from fig_cebra_1 import plot_cebra
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_manif
#from create_h5_store import labels_to_str
#from create_h5_store import generate_group_name
#from cr_db_sql import create_database
#from cr_db_sql import save_fig
#from cr_db_sql import save_manif

#from FIG2_mod import  Fig2_rat_hip
# Now you can call run_hip_models() in your script

def main(params):
    #create_database() 
    base_path=main_path
      

   # Fig2_rat_hip(dd, err_loss, mod_pred,base_path) 
    manif, labels = run_hip_models(base_path,params)
    #unique_name = "manif_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    #save_manif(manif, unique_name)
    #fig=plot_cebra(manif, labels)
    #save_fig(fig, fig_id="my_plot_id")
    file_name = "manif_data.hdf5"

    ### create a group of manifold per label
    group_name= 'Cebra_behav'
    include_labels = True  # vel False, sup vs unsup
    labels = labels  # o le tue labels, se hai deciso di includerle

    
    with create_or_open_hdf5(file_name) as hdf5_file:
        save_manif(hdf5_file, group_name, manif, labels=labels,
        include_labels=include_labels)
    
  

    # Opional: plotting data
    # fig=plot_cebra(manif, labels)
    # save_fig(fig, fig_id="my_plot_id")
    plot_cebra(manif, labels)

    input("Press any key to continue..")




    return manif, labels
    
    #return  dd, err_loss, mod_pred
    
if __name__=="__main__":


    #dd, err_loss, mod_pred= 

    main(params)


file_name = 'manif_data.hdf5'
try:
    with h5py.File(file_name, 'r') as f:
        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
except Exception as e:
    print(e)



#neur=hip_pos.neural.numpy()

#pippo1=dd['visualization']['hypothesis']
#pippo=dd['visualization']['discovery']
#
