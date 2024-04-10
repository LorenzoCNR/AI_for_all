# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:47:10 2024

@author: zlollo2
"""
#sys.path.insert(0,'/path/to/your/directory')
import os
os.getcwd()
import sys
#### Mettere la directory di interesse la stessa di matlab
from pathlib import Path
#base_dir=r'/home/zlollo/CNR/Cebra_for_all'
#os.chdir(base_dir)
import time

#!pip install ripser
#import ripser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
from scipy.io import loadmat
from scipy.io import savemat
#from dataset import SingleRatDataset  
import sklearn.metrics
import inspect
import torch


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np

def plot_cebra(emb, label):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    # Define colormaps
    #cmap_right = plt.get_cmap('summer') 
    #cmap_left = plt.get_cmap('cool')

    # Normalization based on the entire range of labels
    #norm = plt.Normalize(vmin=label[:,0].min(), vmax=label[:,0].max())
    
    # Check if labels have three columns
    if label.shape[1] == 3:
        idx_left = label[:, 2] == 1
        idx_right = label[:, 1] == 1

        # Apply color mapping based on the first column of the labels
        # Note: The color mapping now directly uses the normed label values
       
         #colors_left =  plt.cm.cool(label[idx_left, 0])
         # colors_right = plt.cm.summer(label[idx_right, 0])


        # Create scatter plots
        # Note: 'c' parameter is now used correctly with color values
        scatter_left = ax.scatter(emb[idx_left, 0], emb[idx_left, 1], emb[idx_left, 2],
                                 c=label[idx_left,0],cmap="cool", s=0.5)
        cbar_left = fig.colorbar(scatter_left, ax=ax, pad=0.1)
        cbar_left.set_label('Left')
        
        scatter_right = ax.scatter(emb[idx_right, 0], emb[idx_right, 1], emb[idx_right, 2],
                                  c=label[idx_right,0],s=0.5)
      
        cbar_right = fig.colorbar(scatter_right, ax=ax, pad=0.1)
        cbar_right.set_label('Right')

    else:
        # Separate data into positive and negative labels
        idx_right = label[:, 0] >= 0
        idx_left = label[:, 0] < 0

        scatter_pos = ax.scatter(emb[idx_right, 0], emb[idx_right, 1], emb[idx_right, 2],
                                 c=label[idx_right,0],   s=0.5, )
        cbar_right = fig.colorbar(scatter_pos, ax=ax, pad=0.1)
        cbar_right.set_label('Right')
        scatter_neg= ax.scatter(emb[idx_left, 0], emb[idx_left, 1], emb[idx_left, 2],
                                 c=label[idx_left,0],cmap="cool", s=0.5)
        cbar_left = fig.colorbar(scatter_neg, ax=ax, pad=0.1)
        cbar_left.set_label('Left')

    ax.set_title("Cebra", fontsize=20)
    

    # Show legends if they exist
    #if 'scatter_right' in locals() or 'scatter_pos' in locals():
    #    ax.legend()

    plt.show()
    return fig