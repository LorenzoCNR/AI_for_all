#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:47:54 2023

@author: zlollo
"""

import os
import sys
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
    "hybrid": "False",
    "time_offsets": "10",
    "seed": "32"
    }



from hip_models_1 import run_hip_models
from fig_cebra_1 import plot_cebra
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_manif
def plot_results (hdf5_path):
    # Opional: plotting data
    with h5py.File(hdf5_path, 'r') as hdf:
        labels           = hdf['/data/labels'][:]
        transformed_data = hdf['/data/manifold'][:]
        plot_cebra(transformed_data, labels)

if __name__ == "__main__":
 
    hdf5_path = sys.argv[1]
    plot_results(hdf5_path)