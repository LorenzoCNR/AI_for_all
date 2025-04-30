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
import json
import yaml
import h5py
import time
from datetime import datetime

from hip_models_fit import run_hip_models_fit
from hip_models_transform import run_hip_models_transform
from fig_cebra import plot_cebra
from data_h5_jl_store import create_or_open_hdf5, save_data, save_manif, save_fig_with_timestamp,save_parameters
matplotlib.use('TkAgg')

def main():
    
    # Load configuration
    with open('config_cebra.yaml', 'r') as file:
        config = yaml.safe_load(file)

    main_path = Path(config['paths']['main_path'])
    data_folder = Path(config['paths']['data_folder'])
    output_folder=main_path
    ### optionally give an alternative output folder
    #output_folder = Path(config['paths']['output_folder'])
    ### check if folder existe
   # output_folder.mkdir(parents=True, exist_ok=True)
### define a folder for images 
    IMAGES_PATH = main_path / "images_std_div_et_imp" 
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)

    # Load model parameters
    model_params = config['model_params']

### STEP 1 Load data from original Achille.jl file

    # Load joblib (original Data)
    jl_path = data_folder / 'achilles.jl'
    data_Achilles = jl.load(jl_path)
    print("Achilles Dict keys:", data_Achilles.keys())

    Achille_neural = data_Achilles['spikes']
    Achille_behav = data_Achilles['position']

    # Database names and group names from config
    f_name = config['hd5_specifics']['db_name']
    gr_name = config['hd5_specifics']['group_name']

### Step 2 store loaded data and parameters in a hd5 file
    # Store Achilles data in HDF5 file
    with create_or_open_hdf5(output_folder / f_name) as hdf5_file:
        neural_data_path = config['hd5_specifics']['neural_data_path']
        save_data(hdf5_file, gr_name, neural_data_path, Achille_neural, labels=[], include_labels=False)
        
        behav_data_path = config['hd5_specifics']['behav_data_path']
        save_data(hdf5_file, gr_name, behav_data_path, Achille_behav, labels=[], include_labels=False)


        save_parameters(hdf5_file, gr_name, config['model_params'])
    # Load data and print parameters
    with h5py.File(output_folder / f_name, 'r') as hdf:
        print(f"Checking existence of datasets and parameters in HDF5 file at '{output_folder / f_name}'.")
        
        # Check for the existence of each dataset within their respective group paths
        if behav_data_path in hdf and neural_data_path in hdf:
            # Load and print behavioral data
            dataset_behav = hdf[behav_data_path]
            rat_behav = dataset_behav[:]
            print("Successfully loaded behavioral data!", rat_behav)
            
            # Load and print neural data
            dataset_neural = hdf[neural_data_path]
            rat_neur = dataset_neural[:]
            print("Successfully loaded neural data!", rat_neur)
        else:
            print(f"One or both datasets not found! '{behav_data_path}', '{neural_data_path}'")

        # Check and print parameters if the group exists
        if gr_name in hdf:
            group = hdf[gr_name]
            if group.attrs:
                loaded_params = {attr: group.attrs[attr] for attr in group.attrs.keys()}
                print("Parameters loaded:")
                for key, value in loaded_params.items():
                    print(f"{key}: {value}")
            else:
                print("No parameters found in the group.")
        else:
            print(f"Group '{gr_name}' not found!")
        # Step 3 fit the model adn save it in the given output folder
    model_fit = run_hip_models_fit(main_path, loaded_params, rat_neur, rat_behav, output_folder)
    
    #à Step 4 transform given data according to saved model.
    manif = run_hip_models_transform(model_fit, Achille_neural)


    # Step 5 Save the transformed data into a separate HDF5 for manifolds
    manif_db = config['hd5_specifics']['db_manif']
    manif_group = config['hd5_specifics']['gr_manif1']
    with create_or_open_hdf5(output_folder / manif_db) as hdf5_file:
        save_manif(hdf5_file, manif_group, manif, labels=[], include_labels=False)
        print(f"Manifold data saved in {manif_group} within {manif_db}.")

    # plots

    fig = plot_cebra(manif, rat_behav)
    save_fig_with_timestamp(fig, "my_plot_id", IMAGES_PATH)
    #end_time = time.time()
    #print(f"Total execution time: {end_time - start_time:.2f} seconds")



if __name__ == "__main__":
    main()



