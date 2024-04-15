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



from fig_cebra_1 import plot_cebra

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
    "seed": "32",
    "manifold_filename": "neural.hd5",
    "behavior_filename": "behavior.hd5"
    }
manifold_filename   = "manifold.hdf5"
behavior_filename   = "behavior.hdf5"
manifold_field      = "manifold"
behavior_field      = "behavior"
group_field         = "data"  


def plot_results (hdf5_path):
    # Opional: plotting data
    with h5py.File(behavior_filename, 'r') as hdf:
        behavior    = hdf[group_field + '/' + behavior_field][:]
    with h5py.File(manifold_filename, 'r') as hdf:
        manifold    = hdf[group_field + '/' + manifold_field][:]
    
    plot_cebra(manifold, behavior)

if __name__ == "__main__":
 
    hdf5_path = sys.argv[1]
    plot_results(hdf5_path)