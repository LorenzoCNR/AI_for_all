#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 12:11:11 2024

@author: zlollo
"""

#!pip install ripser
import os
import sys
import warnings
import typing
#os.chdir(os.getcwd())
os.getcwd()
#### windows
base_dir=r'F:\CNR_neuroscience\cebra_git'
## ubuntu
#base_dir=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git'
os.chdir(base_dir)
import argparse
import yaml
import logging
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import openTSNE
import scipy.sparse as sp
import sympy
from joblib import Parallel, delayed
#import ripser
import numpy as np
import torch
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import umap
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
from scipy import stats
# import tensorflow as tf
from cebra import CEBRA
#from dataset import SingleRatDataset  
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
#from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
#import inspect
#import torch
from cebra.datasets.hippocampus import *
import h5py
import pathlib
from matplotlib.markers import MarkerStyle
import seaborn as sns
from scipy.sparse import lil_matrix
import datetime
#from sklearn.model_selection import ParameterGrid



### Ubuntu
input_dir=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes/figure_16_05_2024'
output_dir=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes/figure_16_05_2024'

 # Windows
input_dir = r'F:\CNR_neuroscience\cebra_git\Cebra_for_all\cebra_codes\figure_16_05_2024'
output_dir = r'F:\CNR_neuroscience\cebra_git\Cebra_for_all\cebra_codes\figure_16_05_2024'


output_file='manifold_1.hdf5'
path_to_output=os.path.join(input_dir, output_file)

off_r = 5
off_l = 5
split_no=0
rat_name='achilles'
def load_dataset_for_rat(rat, split_no, split_type='all', seed=None):
    
    """
    Args:
        rat (str)
        split_no (int)
        split_type (str): Tipo di split, default è 'all'.
        seed (int): optional seed; default none. 

    """
    dataset_name = f'rat-hippocampus-{rat}-3fold-trial-split-{split_no}'
    dataset = cebra.datasets.init(dataset_name, split=split_type)

    # Configure offset 
    if hasattr(dataset, 'offset'):
        dataset.offset.right = off_r
        dataset.offset.left = off_l


    return dataset

data_ = load_dataset_for_rat(rat_name, split_no)

X=data_.neural.numpy()
y=data_.continuous_index.numpy()

   
f_name=path_to_output

#pd.read_hdf(f_name, key='cebra_time')
#
#### print group and datasets name
# def print_names(name, obj):
#     print(name)
#     if isinstance(obj, h5py.Group):
#         print(f"{name} is a group")
#     elif isinstance(obj, h5py.Dataset):
#         print(f"{name} is a dataset")
#     else:
#         print(f"{name} Unknown type")
 
# with h5py.File(f_name, 'r') as file:
#     file.visititems(print_names) 
 
### store datasets and group names (just to copy an paste)
def get_dataset_names(file_path):
    with h5py.File(file_path, "r") as h5f:
        groups = list(h5f.keys())
        dataset_names = {group: list(h5f[group].keys()) for group in groups}
    return dataset_names       

dataset_names = get_dataset_names(f_name)
print(dataset_names)          

def rename_dataset(hdf_file, group_name, old_name, new_name):
    
    #Rename datasets in hd5
    
    group = hdf_file[group_name]
    
    if old_name in group:
        # copy data from existing dataset
        if new_name in group:
            print(f"Skipping: Dataset '{new_name}' already exists in group '{group_name}'.")
            return False  #
            
            
        old_dataset = group[old_name]
        data = np.array(old_dataset)
        
        # Create new dataset with new name
        group.create_dataset(new_name, data=data)
        
        # Copy attributes from existing dataset
        for attr_name, attr_value in old_dataset.attrs.items():
            group[new_name].attrs[attr_name] = attr_value
        
        # Delete original and existing dataset
        del group[old_name]
        
        print(f"Dataset '{old_name}' renamed in '{new_name}' in group '{group_name}'")
    else:
        print(f"Dataset '{old_name}' not found in group '{group_name}'")

def generate_new_name(old_name, replacements):

    new_name = old_name
    for old, new in replacements.items():
        new_name = new_name.replace(old, new)
    return new_name


def rename_all_datasets_in_group(hdf_file_path, group_name, replacements):
    
    #Renbame all datasets in group.

    
    new_dataset_names = []
    with h5py.File(hdf_file_path, 'a') as hdf_file:
        group = hdf_file[group_name]
        for old_name in list(group.keys()):
            new_name = generate_new_name(old_name, replacements)
            success=rename_dataset(hdf_file, group_name, old_name, new_name)
            if success:
                new_dataset_names.append(new_name)
    return new_dataset_names

# Dict of replacemnent word
replacements = {
    "learning_rate": "l_r",
    "num_hidden_units": "n_h_u",
    "temperature": "temp",
    "early_exaggeration": "earl_exag",
    "perplexity": "perp"
}


group_name = 'cebra_time'
group_name = 'cebra_behavior'
group_name= 'cebra_hybrid'
group_name= 'tsne'
group_name= 'umap'
group_name= 'conv_pivae'
new_dataset_names = rename_all_datasets_in_group(f_name, group_name, replacements)

# Also
group_names = ['cebra_time', 'cebra_behavior', 'cebra_hybrid', 'tsne', 'umap', 'conv_pivae']

# Rename all dataset in groups'group names
for group in group_names:
    print(f"Renaming datasets in group: {group}")
    new_dataset_names = rename_all_datasets_in_group(f_name, group, replacements)
    print("New names:", new_dataset_names)

### print changes happened
print("New names:", new_dataset_names)



# List
# chosen_datasets = [
#     'cebra_time_20240516_learning_rate0.0003_num_hidden_units32_temperature1',
#     'cebra_time_20240516_learning_rate0.0003_num_hidden_units32_temperature1.5',
#     'cebra_time_20240516_learning_rate0.0003_num_hidden_units32_temperature2',
#     'cebra_time_20240516_learning_rate0.0003_num_hidden_units32_temperature2.5',
#     'cebra_time_20240516_learning_rate0.0003_num_hidden_units32_temperature3'
# ]




### extract data
# 
def extract_datasets_from_group(hdf_file, group_name, dataset_names=None):
    group = hdf_file[group_name]
    data_dict = {}
    
    if dataset_names is None:
        # EXtract all data
        dataset_names = group.keys()
    
    for dataset_name in dataset_names:
        if dataset_name in group:
            dataset = group[dataset_name]
            data_dict[dataset_name] = np.array(dataset)
            print(f' Dataset {dataset_name} shape {data_dict[dataset_name].shape}')
        else:
            print(f'Dataset {dataset_name} not found in {group_name}')
    
    return data_dict



with h5py.File(f_name, 'r') as hdf_file:
    
    group_name = 'cebra_time'
    
    #all_data = extract_datasets_from_group(hdf_file, group_name)
    cebra_time_dict= extract_datasets_from_group(hdf_file, group_name)

def plot_datasets_in_groups(dataset_dict, label,  group_size):
    dataset_names = list(dataset_dict.keys())
    num_datasets = len(dataset_names)
    figures = [] 
    
    
    
    
    for i in range(0, num_datasets, group_size):
        # Compute # subplot
        actual_group_size = min(group_size, num_datasets - i)
        fig, axs = plt.subplots(1, actual_group_size, figsize=(24, 6), subplot_kw={'projection': '3d'})
        if actual_group_size == 1:
            axs = [axs]  # 
        for j in range(actual_group_size):
            dataset_name = dataset_names[i + j]
            emb = dataset_dict[dataset_name]
            
            idx_left = label[:, 2] == 1
            idx_right = label[:, 1] == 1

            scatter_left = axs[j].scatter(emb[idx_left, 0], emb[idx_left, 1], emb[idx_left, 2], c=label[idx_left, 0], cmap="cool_r", s=0.5)
            scatter_right = axs[j].scatter(emb[idx_right, 0], emb[idx_right, 1], emb[idx_right, 2], c=label[idx_right, 0], cmap="summer_r", s=0.5)

            axs[j].axis("off")
            axs[j].set_title(dataset_name)

            # Colorbar
        cbar_left = fig.colorbar(scatter_left, ax=axs, pad=0.02, fraction=0.02, location='bottom', shrink=0.5)
        cbar_left.set_label('Left')
        cbar_right = fig.colorbar(scatter_right, ax=axs, pad=0.02, fraction=0.02, location='bottom', shrink=0.5)
        cbar_right.set_label('Right')

        
        cbar_left.ax.set_position([0.24, 0.05, 0.35, 0.03])  # [x, y, width, height]
        cbar_right.ax.set_position([0.41, 0.05, 0.35, 0.03])
        
                
        plt.subplots_adjust(left=0.1, wspace=0.01)
               
        plt.show()
        figures.append(fig) 
    return figures
       
      # plt.show()




def save_fig(fig_id,filename, tight_layout=True, fig_extension="png", resolution=300):
     #path = output_dir
     if tight_layout:
        try:
            fig.tight_layout()
        except Exception as e:
            print("Warning: Tight layout was not applied due to: ", e)
           
     fig_path = f"{output_dir}/{filename}.{fig_extension}"  
     fig.savefig(fig_path, format=fig_extension, dpi=resolution)
     print(f"Figure saved as {fig_path}")
    # plt.savefig(path, format=fig_extension, dpi=resolution)

#plot_datasets_in_groups(cebra_behav_dict,y, group_size=4)
figures = plot_datasets_in_groups(cebra_time_dict, y, group_size=4)
for i, fig in enumerate(figures):
    save_fig(fig, f"cebra_time_fig{i+1}", output_dir)
    plt.close(fig) 
