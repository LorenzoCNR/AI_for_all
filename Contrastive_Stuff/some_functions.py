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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import re
import joblib as jl
import logging
import time
import scipy.sparse as sp
import sympy
import plotly.graph_objects as go
from sklearn.model_selection import ParameterGrid, train_test_split
#from joblib import Parallel, delayed
import torch
from pathlib import Path
#import multiprocessing
from numpy.lib.stride_tricks import as_strided
#from multiprocessing import Pool
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from io import BytesIO
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
import json
import copy

#plt.switch_backend("webagg")  # Forza il backend del browser
#lt.ion()

#### funzione per settare i  percorsi
############### funzione per ottenere ed impostare i percorsi
#### just notice that some folders are case specific (i.e. monkey data folder)
# Set up a logger

def setup_paths(data_dir,sub_data,out_dir,pipe_path,change_dir=False):
    """
    Configure paths for the project.
    
    Args:
        change_dir (bool): If True, changes the current working 
        directory to the project root.
    
    Returns:
        tuple: project_root, eeg_pipeline_path, default_output_dir,
        default_input_dir
    """
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Determine the project root dynamically
    try:
        project_root = Path(__file__).resolve().parent
    except NameError:
        # Fallback to cwd if __file__ is not available
        project_root = Path(os.getcwd()) 
    logger.info(f"Project root resolved to: {project_root}")

    # Define  paths
    eeg_pipeline_path = project_root / pipe_path
    logger.info(f"Path to 'EEG-ANN-Pipeline': {eeg_pipeline_path}")

    default_input_dir = project_root.parent / data_dir / sub_data
    logger.info(f"Input data path: {default_input_dir}")

    default_output_dir = project_root / out_dir
    logger.info(f"Output path: {default_output_dir}")

    # Add "EEG-ANN-Pipeline" to sys.path
    if str(eeg_pipeline_path) not in sys.path:
        sys.path.append(str(eeg_pipeline_path))
        logger.info(f"Added '{eeg_pipeline_path}' to sys.path")

    # Create output directory if it doesn't exist
    default_output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory ensured: {default_output_dir}")

    # Optionally change the working directory
    if change_dir:
        os.chdir(project_root)
        logger.info(f"Working directory changed to: {project_root}")

    return project_root, eeg_pipeline_path, default_output_dir, default_input_dir









########################### GENERIC LOAD DATA FUNCTION ##############
# def load_data(input_dir, name):
#     input_dir = Path(input_dir).resolve()
#     path =input_dir / f"{name}.jl"
#     print(path)
#     #path = os.path.join(input_dir, f"{name}.jl")
#     try:
#         data = jl.load(path)
#         return data  
#     except FileNotFoundError as e:
#         print(f"File Not Found Error: {e}")
#         return None  # Return



def load_data(input_dir, name, file_format):
    """
    Generic fucntions to load data with different extension.

    Args:
        input_dir (str): Data Folder
        name (str): filename (with no extension).
        file_format (str): file format 
        (es: "json", "pkl", "mat", "jl", "csv", "txt").
    
    Returns:
        Data or none if error.
    """
    input_dir = Path(input_dir).resolve()    
    path = input_dir / f"{name}.{file_format}"  
    
    print(f"try to load: {path}")  # Debug

    try:
        if file_format == "json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        
        elif file_format == "pkl":  # Pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        
        elif file_format == "jl":  # Joblib
            return jl.load(path)
        
        elif file_format == "mat":  # MATLAB .mat
            return scipy.io.loadmat(path)
        
        elif file_format == "csv":  # CSV (DataFrame)
            return pd.read_csv(path)
        
        elif file_format == "txt":  # Pure Text
            with open(path, "r", encoding="utf-8") as f:
                return f.readlines()
        
        else:
            print(f"unsupported'{file_format}' format.")
            return None
    
    except FileNotFoundError:
        print(f"Error: File '{path}' not found.")
        return None
    
    except Exception as e:
        print(f"Error while loading: '{path}': {e}")
        return None






###################### CREATE TRIALS ACCORDING TO BEHAVIOR ####################

       

def create_rats_trial(behav_data):
    """Create trial identifiers based on behavioral data and track start
    positions of new trials.
    VALID FOR CEBRA RATS (Buszaki 2015)"""
    trial_ids = np.zeros(len(behav_data), dtype=int)  # Initialize trial IDs with 0s
    c_t = [0]  # Start position of the first trial
    current_trial = 1
    trial_ids[0] = current_trial  # Start first trial with ID 1
    change_count = 0  # Counter for changes in behavior data
    
    for i in range(1, len(behav_data)):
        if behav_data[i] != behav_data[i - 1]:
            change_count += 1  # Increment change counter on data change
            
            if change_count == 2:
                current_trial += 1  # Increment trial ID every two changes
                c_t.append(i)  # Append start index of new trial
                change_count = 0  # Reset change counter
        
        trial_ids[i] = current_trial  # Assign current trial ID to each position
        
    return trial_ids, c_t 

#### RESAMPLIGN DATA FUNCTION 
#### RESAMPLIGN DATA FUNCTION 
def f_resample(datasets, trials, step, overlap, methods, mode="disjoint", normalization=False):  
    """
    
    resampling of dataset list, split into trial taking with the option to 
    take overlapping or disjoint windows
    - window central value
    - window mean
    - window sum
    
    Args:
        - datasets (lis):dataset to be resampled
        - trials (list): every sublist tells where every trial starts and ends
        - step (int): resmplign window dimension
        - methods (dict): dictionary with key mean center or sum
        - overlap (int): Numero di punti di sovrapposizione tra finestre
            consecutive (on only with mode="overlapping").
        mode (str):  resampling mode, "disjoint" (no overlapping) 
        "overlapping" (moving window)

    Returns:
        list: Lista di dataset resamplati.
        new_trial_lengths: list of lengths of each resampled trial
        new_trials_indices: list of (start, end) indices for each trial
    
    """
    resampled_datasets = []
    new_trials_indices = []
    new_trial_lengths = []

    stride = step if mode == "disjoint" else step - overlap  

    for i, dataset in enumerate(datasets):
        resampled_trials = []
        trial_lengths = []
        trial_indices = []
        # if not specified default method is "center"
        method = methods.get(i, "center")  
        current_start=0
        # Iterate pver trials of the i-th dataset
        for start_trial, end_trial in trials: 
            # extract trial
            trial_data = dataset[start_trial:end_trial]  
            n_rows = len(trial_data)

            resampled = []
            # actual window
            for start in range(0, n_rows - step + 1, stride):
                end = start + step  

                if method == "center":
                    idx = min(start + step // 2, n_rows - 1)
                    resampled.append(trial_data[idx])
                
                elif method == "mean":
                    resampled.append(np.mean(trial_data[start:end], axis=0))
                
                elif method == "sum":
                    sum_val = np.sum(trial_data[start:end], axis=0)
                    if normalization:
                        # Normalization to prevent inflated counts
                        corrected_sum = sum_val * (stride / step)  
                        resampled.append(corrected_sum)
                    else:
                        resampled.append(sum_val)
                    
                    
              
        

            if resampled:
                resampled_array = np.array(resampled)
                resampled_trials.append(resampled_array)
                trial_len = resampled_array.shape[0]
                trial_lengths.append(trial_len)
                trial_indices.append((current_start, current_start + trial_len-1))
                current_start += trial_len
            else:
                trial_lengths.append(0)
                trial_indices.append((current_start, current_start))

        if resampled_trials:
            resampled_concat = np.concatenate(resampled_trials)
            resampled_datasets.append(resampled_concat)
        else:
            resampled_datasets.append(np.array([]))

        new_trial_lengths.append(trial_lengths)
        new_trials_indices.append(trial_indices)
    return resampled_datasets, new_trial_lengths, new_trials_indices

############################ DATA SPLITTER ACCORDING TO TRIALS   #######################

from sklearn.model_selection import train_test_split
import numpy as np

# Case 1: Train, Validation, and Test
# Case 2: Train and Validation only
# Case 3: Train (Subtrain and Validation within Train) and Test


def split_data_trials(data, case=1, shuffle=False, seed=None, train_ratio=0.7, val_ratio=0.3, verbose=False, update_original=True):
    """
    Splits data into train, validation, and test based on unique trials.
    
    Parameters:
        data (dict): Dictionary containing 'X', 'y', and 'trials'.
        case (int): Split case (1: train/val/test, 2: train/val, 
                                3: train (subtrain+val)/test).
        shuffle (bool): Whether to shuffle the unique trials.
        seed (int): Random seed for reproducibility.
        train_ratio (float): Proportion of trials for training.
        val_ratio (float): Proportion of trials for validation.
        verbose (bool): If True, print split details.
        update_original (bool): If True, updates the input `data` dictionary.
    
    Returns:
        dict: Dictionary with split trials 
        (train, val, test, subtrain if applicable).
    """
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
        print(f"test_ratio is: {test_ratio}")
        if test_ratio <= 0:
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
    else:
        raise ValueError("Invalid case selected. Choose 1, 2, or 3.")
        
    def get_trial_indices(trial_array, trial_numbers):
        return np.isin(trial_array, trial_numbers).nonzero()[0]

    # Get indices
    train_idx = get_trial_indices(trials, train_trials)
    val_idx = get_trial_indices(trials, val_trials)
    test_idx = get_trial_indices(trials, test_trials) if test_trials is not None else None
    subtrain_idx = get_trial_indices(trials, subtrain_trials) if case == 3 else train_idx

    # Prepare splits
    split_data = {
        "X_train": spikes[train_idx],
        "y_train": behavior[train_idx],
        "X_val": spikes[val_idx],
        "y_val": behavior[val_idx],
        "train_trials": train_trials,
        "val_trials": val_trials,
    }

    if test_trials is not None:
        split_data.update({
            "X_test": spikes[test_idx],
            "y_test": behavior[test_idx],
            "test_trials": test_trials,
        })

    if case == 3:
        split_data.update({
            "X_subtrain": spikes[subtrain_idx],
            "y_subtrain": behavior[subtrain_idx],
            "subtrain_trials": subtrain_trials,
        })

    # Update original data if specified
    if update_original:
        data.update(split_data)
        return data
    else:
        return split_data     
        
    # Function to get indices of rows belonging to specific trials
    # def get_trial_indices(trial_array, trial_numbers):
    #     return np.isin(trial_array, trial_numbers).nonzero()[0]

    # # Get indices for each set
    # train_idx = get_trial_indices(trials, train_trials)
    # val_idx = get_trial_indices(trials, val_trials)
    # test_idx = get_trial_indices(trials, test_trials) if test_trials is not None else None
    # subtrain_idx = get_trial_indices(trials, subtrain_trials) if case == 3 else train_idx

    # if verbose:
    #     print(f"Total trials: {total_trials}")
    #     print(f"Train trials: {len(train_trials)}, Validation trials: {len(val_trials)}")
    #     if test_trials is not None:
    #         print(f"Test trials: {len(test_trials)}")
    #     if case == 3:
    #         print(f"Subtrain trials: {len(subtrain_trials)}")
    #     print(f"Train indices: {train_idx}")
    #     print(f"Validation indices: {val_idx}")
    #     if test_trials is not None:
    #         print(f"Test indices: {test_idx}")
    #     if case == 3:
    #         print(f"Subtrain indices: {subtrain_idx}")

    # # deliver data splits  to the data dictionary
    # data['X_train'] = spikes[train_idx]
    # data['y_train'] = behavior[train_idx]
    
    # data['X_val'] = spikes[val_idx]
    # data['y_val'] = behavior[val_idx]

    # data['train_trials']=train_trials
    # data['val_trials']=val_trials
    
    # if test_trials is not None:
    #     data['X_test'] = spikes[test_idx]
    #     data['y_test'] = behavior[test_idx]
    #     data['test_trials']=test_trials

    
    # if case == 3:
    #     data['X_subtrain'] = spikes[subtrain_idx]
    #     data['y_subtrain'] = behavior[subtrain_idx]
    #     data['subtrain_trial'] = [subtrain_trials]
    # return data



##### LOAD PARAMETERS from yaml (actual version fits cebra needings)

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

def load_model_params(param_file, model_type):
    """
    Carica i parametri dal file YAML per il tipo di modello specificato.

    Parameters:
        param_file (str): Path al file YAML.
        model_type (str): Tipo di modello ('cebra_time', 'cebra_behavior', etc.).

    Returns:
        tuple: (fixed_params, grid_params)
    """
    params = load_params(param_file)  # Usa la tua funzione esistente
    
    # Controlla che il modello sia nel file
    if model_type not in params:
        raise ValueError(f"Model type '{model_type}' non trovato in {param_file}.")
    
    fixed_params = params[model_type].get('fixed', {})
    grid_params = params[model_type].get('grid', {})
    
    return fixed_params, grid_params

##################### PLOTS

##### rat_hippocampus

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
            
        ax.axis("off")
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

def plot_embs_continuous(emb, label, title):
    fig = plt.figure(figsize=(12, 5))
    #plt.suptitle('CEBRA-behavior trained with position label',
              #   fontsize=20)
    ax = plt.subplot(121, projection = '3d')
    ax.set_title('x', fontsize=20, y=0)
    x = ax.scatter(emb[:, 0],
                   emb[:, 1],
                   emb[:, 2],
                   c=label[:, 0],
                   cmap='seismic',
                   s=0.05,
                   vmin=-15,
                   vmax=15)
    ax.axis('on')
    ax = plt.subplot(122, projection = '3d')
    y = ax.scatter(emb[:, 0],
                   emb[:, 1],
                   emb[:, 2],
                   c=label[:, 1],
                   cmap='seismic',
                   s=0.05,
                   vmin=-15,
                   vmax=15)
    ax.axis('on')
    ax.set_title('y', fontsize=20, y=0)
    yc = plt.colorbar(y, fraction=0.03, pad=0.05, ticks=np.linspace(-15, 15, 7))
    yc.ax.tick_params(labelsize=15)
    yc.ax.set_title("(cm)", fontsize=10)
    plt.show()


def plot_embs_discrete(emb, label, title,ratio, ww):
    fig = plt.figure(figsize=(12, 5))
    #plt.suptitle('CEBRA-behavior trained with target label',
               #  fontsize=5)
    ax = plt.subplot(121, projection = '3d')
    ax.set_title('All trials embedding', fontsize=10, y=-0.1)
    x = ax.scatter(emb[:, 0],
                   emb[:, 1],
                   emb[:, 2],
                   c=label,
                   cmap=plt.cm.hsv,
                   s=0.01)
    ax.axis('on')

    ax = plt.subplot(122,projection = '3d')
    ax.set_title('direction-averaged embedding', fontsize=10, y=-0.1)
    for i in range(8):
        direction_trial = (label == i)
        trial_avg = emb[direction_trial, :].reshape(-1, ratio-ww,
                                                             3).mean(axis=0)
        trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
        ax.scatter(trial_avg_normed[:, 0],
                   trial_avg_normed[:, 1],
                   trial_avg_normed[:, 2],
                   color=plt.cm.hsv(1 / 8 * i),
                   s=0.2)
    ax.axis('on')
    plt.show()
    
    
    

def plot_direction_averaged_embedding(z_, l_dir_, original_label_order, c_s,
                                      output_folder, name, trial_length=None,
                                      constant_length=True, ww=10,
                                      label_swap_info=None):
    """
  Plot averaged neural embedding trajectories normalized on a unit sphere,
  preserving the original label order and optionally displaying a mapping
  between modified and original labels in the legend.

  Args:
      z_: ndarray - Neural embedding (time x dim)
      l_dir_: ndarray - Labels used during training (possibly remapped)
      original_label_order: list[int] - Order in which to plot original directions
      c_s: str - Color used for "start" markers
      output_folder: str - Path where output will be saved
      name: str - Name of the HTML file to export
      trial_length: int - Length of each trial (required for reshaping)
      constant_length: bool - If True, assumes trial lengths are uniform
      ww: int - Model receptive window size to subtract
      label_swap_info: dict - Mapping {original_label: label_used_in_training}
  """
    
    # create unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    fig = go.Figure()
    fig.add_trace(go.Surface(x=x, y=y, z=z, colorscale="Blues", opacity=0.1, 
                             showscale=False))
    
    
    #   "Start" and "End" markers
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(color=c_s, size=5, symbol="circle"),
        name="Start"
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode="markers",
        marker=dict(color="black", size=5, symbol="x"),
        name="End"
    ))
    # Unique directions
    unique_dirs = np.unique(l_dir_)
    n_colors = len(unique_dirs)

    # Setup colormap
    cmap = plt.get_cmap('hsv', n_colors)
    norm = mcolors.Normalize(vmin=0, vmax=n_colors - 1)

   

    # Define 2D circle positions for each direction (for visual reference)
    # radius = 1.2
    # direction_positions = [(radius * np.cos(np.deg2rad(360 * i / n_colors)),
    #                         radius * np.sin(np.deg2rad(360 * i / n_colors)))
    #                        for i in range(n_colors)]
    # for i, (x, y) in enumerate(direction_positions):
    #     fig.add_trace(go.Scatter3d(
    #         x=[x], y=[y], z=[0],
    #         mode="text",
    #         text=[f"Dir {i+1}"],
    #         textposition="middle center",
    #         showlegend=False
    #     ))

    # Loop over trajectories
    for idx, original_label in enumerate(original_label_order):

    # Map the original label to the actual one used during training
        used_label = label_swap_info.get(original_label, original_label) if label_swap_info else original_label
        mask = (l_dir_ == original_label)
        print(mask.shape)
        if not np.any(mask):
            print(f" No data found for original label {original_label} (used {used_label})")
            continue
    
        try:
           trial_avg = z_[mask].reshape(-1, trial_length - ww, 3).mean(axis=0)
        except ValueError as e:
             print(f"ValueError for label {original_label}: {e}")
             continue

    # Normalize each vector (to lie on the unit sphere)
       
        print("Trial average shape:", trial_avg.shape)
        print(trial_avg.shape)
        #trial_avg -= trial_avg.mean(axis=0)  # Sottraggo la media di ogni coordinata
        trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    #     #trial_avg /= np.linalg.norm(trial_avg, axis=2, keepdims=True)  # Normalizzazione
    #    # trial_avg_normed = trial_avg.mean(axis=0) 
    #     #trial_avg = z_[direction_trial, :].reshape(-1, trial_length - ww, 3).mean(axis=0)
    #     #trial_avg_normed = trial_avg / np.linalg.norm(trial_avg, axis=1)[:, None]
    
        # Colore della traiettoria con Matplotlib colormap
        #color =cmap(idx)   # Stesso colore di Matplotlib
        

        
        
        # Map color to the current trajectory
        color = cmap(norm(idx))
        #color = cmap(norm(idx - 1))
        hex_color = f'rgb({color[0]*255:.0f},{color[1]*255:.0f},{color[2]*255:.0f})'
        
        
                # Construct label for the legend
        if label_swap_info and original_label in label_swap_info:
            label_display = f"{original_label} → {label_swap_info[original_label]}"
        else:
            label_display = f"{original_label}"
        
                
        # Aggiunta della traiettoria
        fig.add_trace(go.Scatter3d(
            x=trial_avg_normed[:, 0],
            y=trial_avg_normed[:, 1],
            z=trial_avg_normed[:, 2],
            mode="lines",
            line=dict(color=hex_color, width=3),
            # direction label
            name=label_display

        ))
    
        # Aggiunta dei marker Start e End per OGNI traiettoria
        fig.add_trace(go.Scatter3d(
            x=[trial_avg_normed[0, 0]],
            y=[trial_avg_normed[0, 1]],
            z=[trial_avg_normed[0, 2]],
            mode="markers+text",
            marker=dict(color=c_s, size=3, symbol="circle"),
            text=[f"s {original_label}"],
            textposition="top right",
            showlegend=False   
        ))
    
        fig.add_trace(go.Scatter3d(
            x=[trial_avg_normed[-1, 0]],
            y=[trial_avg_normed[-1, 1]],
            z=[trial_avg_normed[-1, 2]],
            mode="markers+text",
            marker=dict(color="black", size=3, symbol="x"),
            text=[f"e {original_label}"],
            textposition="top right",
            showlegend=False  
        ))
    
    # Griglia e legenda
    fig.update_layout(
        title=dict(
            ## chekc the name
        text=name,  
        x=0.5,  #
        xanchor='center',  
    ),
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor="gray", title='x1'),
            yaxis=dict(showgrid=True, gridcolor="gray", title='x2'),
            zaxis=dict(showgrid=True, gridcolor="gray", title='x3'),
        ),
        legend=dict(x=1.1, y=1, font=dict(size=10), title ='Direction of Movement'),
    )
    
    # Salvataggio del file HTML e PNG
    #output_folder = "output_plots"
    #os.makedirs(output_folder, exist_ok=True)
    output_html = os.path.join(output_folder, name)
    #output_png = os.path.join(output_folder, "plot_interattivo.png")
    
    fig.write_html(output_html)
    #fig.write_image(output_png, scale=2)
    fig.show(renderer="browser")



############################# KNN DECODER ####################################à

        
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

   test_score = sklearn.metrics.r2_score(
   label_test[:, :2], prediction, multioutput='variance_weighted')
   max_label = np.max(label_test[:, 0])
   print(f'Max value of label_test[:, 0] is: {max_label}')
    
   pos_test_err_perc = np.median(abs(prediction[:, 0] - label_test[:, 0]) / max_label * 100)
    
   pos_test_err = np.median(abs(prediction[:, 0] - label_test[:, 0]))
    
   pos_test_score = sklearn.metrics.r2_score(
        label_test[:, 0], prediction[:, 0])

   return test_score, pos_test_err,pos_test_err_perc, pos_test_score



   ##### esplora contenuto oggetto ###3

# def explore_obj(obj):
#     """
#     Esplora un oggetto Python, elencando gli attributi e i loro valori.
    
#     Args:
#         obj: Oggetto da esplorare.
        
#     Output:
#         Stampa gli attributi e i loro valori.
#     """
#     print(f"Esplorazione dell'oggetto: {obj.__class__.__name__}")
#     print("-" * 50)
    
#     for attr in dir(obj):
#         # Ignora gli attributi speciali (che iniziano e finiscono con __)
#         if not attr.startswith("__"):
#             try:
#                 valore = getattr(obj, attr)
#                 print(f"{attr}: {valore}")
#             except Exception as e:
#                 print(f"{attr}: (non accessibile, errore: {e})")
#     print("-" * 50)
    
def explore_obj(obj):
    """
    Esplora un oggetto Python, elencando gli attributi e i loro valori,
    inclusa la dimensione o la lunghezza quando pertinente.
    
    Args:
        obj: Oggetto da esplorare.
        
    Output:
        Stampa gli attributi e i loro valori, con dimensioni o lunghezze se applicabile.
    """
    print(f"Esplorazione dell'oggetto: {obj.__class__.__name__}")
    ### linea di apertura e di chiusura
    print("-" * 50)
    
    for attr in dir(obj):
        # Ignora gli attributi speciali (che iniziano e finiscono con __)
        if not attr.startswith("__"):
            try:
                value = getattr(obj, attr)
                # Gestione di diversi tipi di dati
                ### numpy
                if isinstance(value, np.ndarray):
                    print(f"{attr}: array shape {value.shape}")
                ### lista o tupla
                elif isinstance(value, (list, tuple)):
                    print(f"{attr}: length {len(value)}")
                # per tensori PyTorch e altri oggetti simili
                elif hasattr(value, 'shape'):  
                    print(f"{attr}: shape {value.shape}")
                  # per elementi che supportano .size
                elif hasattr(value, 'size'):
                    print(f"{attr}: size {value.size()}")
                else:
                    print(f"{attr}: {value}")
            except Exception as e:
                print(f"{attr}: (non accessibile, errore: {e})")
    print("-" * 50)



# Esempio di utilizzo
#esplora_oggetto(dataset_training)

################# Data Split
### split mode 
def circular_initial_split(trials, train_ratio, seed=None):
    """Initial split train test (will remain out of total process)."""
    np.random.seed(seed)
    unique_trials = np.unique(trials)
    total_trials = len(unique_trials)

    train_count = int(round(train_ratio * total_trials))
    val_count = total_trials - train_count

    # Scegli un punto di partenza casuale
    start_idx = np.random.randint(0, total_trials)
    
    # Indici circolari per train e validation
    train_indices = (np.arange(start_idx, start_idx + train_count) % total_trials)
    val_indices = (np.arange(start_idx + train_count, start_idx + train_count + val_count) % total_trials)

    train_trials = unique_trials[train_indices]
    val_trials = unique_trials[val_indices]

    return train_trials, val_trials




## (circular) cross valdiation function
def circular_cv_split(trials, train_trials, val_ratio, split_idx):
    """Perform a circualr split on train set to cross validate."""
    unique_train_trials = np.unique(train_trials)
    total_train_trials = len(unique_train_trials)

    val_count = int(round(val_ratio * total_train_trials))
    subtrain_count = total_train_trials - val_count

    #  Circular split for cross-validation
    start_idx = (split_idx * val_count) % total_train_trials
    subtrain_indices = (np.arange(start_idx, start_idx + subtrain_count) % total_train_trials)
    val_indices = (np.arange(start_idx + subtrain_count, start_idx + subtrain_count + val_count) % total_train_trials)

    subtrain_trials = unique_train_trials[subtrain_indices]
    val_trials = unique_train_trials[val_indices]

    return subtrain_trials, val_trials

###################################### LABEL SWAP ############################

def swap_labels(labels, swap_dict):
    """
    Swap lables accordin to a given dict

    Args:
        labels (np.array): original labels
        swap_dict (dict): the given dict to swap, es. {3: 6, 6: 3}.

    Returns:
        np.array: swapped array
            """
    swapped_labels = labels.copy()
    for original, new in swap_dict.items():
        swapped_labels[labels == original] = new
    return swapped_labels