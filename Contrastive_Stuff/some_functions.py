# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:53:23 2024

@author: loren
"""

import os
import sys
import warnings
import typing
import numpy as np
import yaml
import argparse
import re
import joblib as jl
import logging
import time
#import openTSNE
import argparse
import scipy.sparse as sp
import sympy
from sklearn.model_selection import ParameterGrid, train_test_split
from joblib import Parallel, delayed
import numpy as np
import torch
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#import umap
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from cebra import CEBRA
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from cebra.datasets.hippocampus import *
import h5py
import pathlib
from matplotlib.markers import MarkerStyle
import seaborn as sns
import pickle
import hashlib
import json
import joblib
import copy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#plt.switch_backend("webagg")  # Forza il backend del browser
#lt.ion()
###################### CREATE TRIALS ACCORDING TO BEHAVIOR ####################

def create_trial_ids(behav_data):
        """Create trial identifiers based on behavioral data and track start positions of new trials."""
        trial_ids = np.ones(len(behav_data), dtype=int)  # Initialize trial IDs with 1s
        c_t = [0]  # Start position of the first trial
        current_trial = 1
        
        for i in range(1, len(behav_data)):
            # Check if behavior data has changed
            if behav_data[i] != behav_data[i - 1]: 
                trial_ids[i] = 2  # Mark as a change for clarity (optional)
                
                # Detect the beginning of a new trial (e.g., round trip start)
                if behav_data[i - 1] == 0 and behav_data[i] == 1:
                    current_trial += 1
                    c_t.append(i)  # Append index where new trial starts
            
            # Assign the current trial ID
            trial_ids[i] = current_trial
        
        return trial_ids, c_t
    


############################ DATA SPLITTER ACCORDING TO TRIALS   #######################

from sklearn.model_selection import train_test_split
import numpy as np

# Case 1: Train, Validation, and Test
# Case 2: Train and Validation only
# Case 3: Train (Subtrain and Validation within Train) and Test


def split_data_trials(data, case=1, shuffle=False, seed=None, train_ratio=0.7, val_ratio=0.3, verbose=False):

    spikes = data['X']
    behavior = data['y']
    trials = data['trials']

    unique_trials = np.unique(trials)

    # Shuffle unique trials if specified
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(unique_trials)

    total_trials = len(unique_trials)

    # Case 1: Train, Validation, and Test
    if case == 1:
        test_ratio = 1 - train_ratio - val_ratio
        if test_ratio < 0:
            raise ValueError("Train and validation ratios are too large to allow a test split.")
        
        # Calculate exact counts
        train_count = int(round(train_ratio * total_trials))
        val_count = int(round(val_ratio * total_trials))
        test_count = total_trials - train_count - val_count  # Ensures the total adds up correctly

        # match trials obs
        train_trials = unique_trials[:train_count]
        val_trials = unique_trials[train_count:train_count + val_count]
        test_trials = unique_trials[train_count + val_count:] if test_count > 0 else None

    # Case 2: Train and Validation only
    elif case == 2:
        train_count = int(round(train_ratio * total_trials))
        val_count = total_trials - train_count

        # Sequential split for Train and Validation
        train_trials = unique_trials[:train_count]
        val_trials = unique_trials[train_count:]
        test_trials = None

    # Case 3: Train (Subtrain and Validation within Train) and Test
    elif case == 3:
        test_ratio = 1 - train_ratio
        if test_ratio < 0:
            raise ValueError("Train ratio is too large to allow a test split.")
        
        
        train_count = int(round(train_ratio * total_trials))
        test_count = total_trials - train_count

        # match trials
        train_trials = unique_trials[:train_count]
        test_trials = unique_trials[train_count:] if test_count > 0 else None
        
        # Further split within Train for Subtrain and Validation
        subtrain_count = int(round((1 - val_ratio) * len(train_trials)))
        subtrain_trials = train_trials[:subtrain_count]
        val_trials = train_trials[subtrain_count:]

    # Function to get indices of rows belonging to specific trials
    def get_trial_indices(trial_array, trial_numbers):
        return np.isin(trial_array, trial_numbers).nonzero()[0]

    # Get indices for each set
    train_idx = get_trial_indices(trials, train_trials)
    val_idx = get_trial_indices(trials, val_trials)
    test_idx = get_trial_indices(trials, test_trials) if test_trials is not None else None
    subtrain_idx = get_trial_indices(trials, subtrain_trials) if case == 3 else train_idx

    if verbose:
        print(f"Total trials: {total_trials}")
        print(f"Train trials: {len(train_trials)}, Validation trials: {len(val_trials)}")
        if test_trials is not None:
            print(f"Test trials: {len(test_trials)}")
        if case == 3:
            print(f"Subtrain trials: {len(subtrain_trials)}")
        print(f"Train indices: {train_idx}")
        print(f"Validation indices: {val_idx}")
        if test_trials is not None:
            print(f"Test indices: {test_idx}")
        if case == 3:
            print(f"Subtrain indices: {subtrain_idx}")

    # deliver data splits  to the data dictionary
    data['X_train'] = spikes[train_idx]
    data['y_train'] = behavior[train_idx]
    
    data['X_val'] = spikes[val_idx]
    data['y_val'] = behavior[val_idx]

    data['train_trials']=train_trials
    data['val_trials']=val_trials
    
    if test_trials is not None:
        data['X_test'] = spikes[test_idx]
        data['y_test'] = behavior[test_idx]
        data['test_trials']=test_trials

    
    if case == 3:
        data['X_subtrain'] = spikes[subtrain_idx]
        data['y_subtrain'] = behavior[subtrain_idx]
        data['subtrain_trial'] = [subtrain_trials]
    return data



##### load parameters

extra_params = {}
def load_params(params_path):
    # check existence
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"file not found{params_path}")
    
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(f"Error in reading from yaml: {exc}")
        return None  
    #except Exception as exc:
     #   print(f"Error!! {exc}")
     #   return None

    return params


##################### PLOTS

def plot_embs(emb,label, title):
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, projection="3d")
        cmap1 = plt.get_cmap('cool')
        cmap2 = plt.get_cmap('summer')
        norm = plt.Normalize(vmin=label[:, 0].min(), vmax=label[:, 0].max())
        
        r = label[:, 1] == 1
        l = label[:, 2] == 1
         
        # Plot per sinistra e destra
        ax.scatter(emb[l, 0], emb[l, 1], emb[l, 2], c=label[l, 0], cmap=cmap1, norm=norm, s=1, label='Left')
        ax.scatter(emb[r, 0], emb[r, 1], emb[r, 2], c=label[r, 0], cmap=cmap2, norm=norm, s=1, label='Right')
            
        ax.axis("on")
        plt.title(title)

        sm_l = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
        #sm_l.set_array([])
        cbar_l= plt.colorbar(sm_l, ax=fig.axes, orientation='vertical', fraction=0.02, pad=0.1)
        cbar_l.set_label('left')

        sm_r = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
        #sm_r.set_array([])
        cbar_r = plt.colorbar(sm_r, ax=fig.axes, orientation='vertical', fraction=0.0176, pad=0.1)
        cbar_r.set_label('right')
        
        
        # horizontal cbar
        #sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
        #sm.set_array([])  # necessaria per l'uso del colorbar
        #cbar = plt.colorbar(sm, ax=fig.axes, orientation='horizontal', fraction=0.02, pad=0.1)
        #cbar.set_label('Position')
       

        #lt.legend()
        plt.show() 

        #### knn decoder

        
def decoding_knn(embedding_train, embedding_test, label_train, label_test):
   metric='cosine'
   n_n=25
   pos_decoder = KNeighborsRegressor(n_neighbors=n_n, metric=metric)
   dir_decoder = KNeighborsClassifier(n_neighbors=n_n, metric=metric)

   pos_decoder.fit(embedding_train, label_train[:,0])
   dir_decoder.fit(embedding_train, label_train[:,1])

   pos_pred = pos_decoder.predict(embedding_test)
   dir_pred = dir_decoder.predict(embedding_test)

   prediction = np.stack([pos_pred, dir_pred],axis = 1)

   test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction, multioutput='variance_weighted')
   pos_test_err = np.median(abs(prediction[:,0] - label_test[:, 0]))
   pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])

   return test_score, pos_test_err, pos_test_score