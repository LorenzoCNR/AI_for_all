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
def plot_results (hdf5_path):
    # Opional: plotting data
    with h5py.File(hdf5_path, 'r') as hdf:
        labels           = hdf['/data/behavior'][:]
        transformed_data = hdf['/data/manifold'][:]
        plot_cebra(transformed_data, labels)

if __name__ == "__main__":
 
    hdf5_path = sys.argv[1]
    plot_results(hdf5_path)