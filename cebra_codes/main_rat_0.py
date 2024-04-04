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
import joblib as jl
from matplotlib.collections import LineCollection
import inspect
import torch
import tensorflow as tf
##### cambiare eventualmente


main_path=r'/media/zlollo/STRILA/CNR neuroscience/cebra_codes'
os.chdir(main_path)
#main_path=r'/home/zlollo/CNR/Cebra_for_all'

#os.chdir(main_path)

os.getcwd()
#from pathlib import Path

# ### crea una catella per immagini qualora non c
# IMAGES_PATH = Path() / "images" 
# IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    
# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#         path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
#         if tight_layout:
#             plt.tight_layout()
#             plt.savefig(path, format=fig_extension, dpi=resolution)
  
  
'''
I parametri di default si cambiano o nel file mat o nel file
hip_models_0 (ci sta un'eccezione con parametri di  default
              qualora non si trovasse il params.mat' 
              verso riga 100)


'''


from hip_models_0 import run_hip_models
from fig_cebra import plot_cebra
#from FIG2_mod import  Fig2_rat_hip
# Now you can call run_hip_models() in your script

def main():
    base_path=main_path
    manif = run_hip_models(base_path)     
   # Fig2_rat_hip(dd, err_loss, mod_pred,base_path) 
    
    #return  dd, err_loss, mod_pred
    return manif
if __name__=="__main__":
    #dd, err_loss, mod_pred= 
     manif, labels=main()


plot_cebra(manif, labels)


#neur=hip_pos.neural.numpy()

#pippo1=dd['visualization']['hypothesis']
#pippo=dd['visualization']['discovery']
#