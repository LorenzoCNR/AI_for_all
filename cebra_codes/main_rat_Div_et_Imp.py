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
import matplotlib
import joblib as jl


matplotlib.use('TkAgg')

from matplotlib.collections import LineCollection
import h5py
#import tensorflow as tf
from pathlib import Path
from datetime import datetime
from hip_models_fit import run_hip_models_fit
from hip_models_transform import run_hip_models_transform

#from-
from fig_cebra import plot_cebra
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_manif
from  create_h5_store import save_fig_with_timestamp

### run hip models 

# Now you can call run_hip_models() in your script
# Provide parameters at the top of the code
# Provide data within the main function from folder
# Just change the group name if different purpose
# image folder name (line 48)

##### cambiare eventualmente

main_path_str=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'

main_path = Path(main_path_str)
os.chdir(main_path)
#main_path=r'/home/zlollo/CNR/Cebra_for_all'

### data
rat_neur=np.load('rat_neural.npy')
#rat_behav=np.load('rat_behaviour_std.npy')
rat_behav=np.load('rat_behaviour_mod.npy')


### h5df name of file and group 
### just change the grpup name 
f_name="manif_file_0.hdf5"
gr_name='manif_mod_div_et_imp'
####  image folder
IMAGES_PATH = main_path / "images_mod_div_et_imp" 
IMAGES_PATH.mkdir(parents=True, exist_ok=True)

os.chdir(main_path)

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
    "max_iterations": "5000",
    "distance": "cosine",
    "conditional": "time_delta",
    "hybrid": "False",
    "time_offsets": "10"
    }



#from create_h5_store import labels_to_str
#from create_h5_store import generate_group_name
#from cr_db_sql import create_database
#from cr_db_sql import save_fig
#from cr_db_sql import save_manif




def main(params):
    #create_database() 
    base_path=main_path
    neural_data=rat_neur
    labels=rat_behav

   # Fig2_rat_hip(dd, err_loss, mod_pred,base_path) 
   # 1) fit and save model
    model_fit = run_hip_models_fit(base_path,params,neural_data, labels)
    # create file to store everythn 
    model_fit_=jl.load('cebra_fit.pkl')

    manif=run_hip_models_transform(model_fit_,base_path, neural_data)
   
    file_name = f_name

    ### create a group of manifold per label or per condition 
    group_name= gr_name
    include_labels = True  # vel False, sup vs unsup
    labels = labels  # o le tue labels, se hai deciso di includerle

    
    with create_or_open_hdf5(file_name) as hdf5_file:
        save_manif(hdf5_file,group_name,manif,labels=labels,include_labels=include_labels)
    
  

    # Opional: plotting data
    fig=plot_cebra(manif, labels)
    ## Save figure in the given path (cfr line 30-40)
    save_fig_with_timestamp(fig, "my_plot_id", IMAGES_PATH)
    #plot_cebra(manif, labels)

    input("Press any key to continue..")


   # return manif
   # 
    #return  dd, err_loss, mod_pred
    
if __name__=="__main__":


    #dd, err_loss, mod_pred= 

    main(params)



