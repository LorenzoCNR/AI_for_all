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
import matplotlib

matplotlib.use('TkAgg')

import joblib as jl
from matplotlib.collections import LineCollection
import inspect

import torch
import tensorflow as tf
from pathlib import Path
from datetime import datetime
##### cambiare eventualmente


main_path=r'/media/zlollo/STRILA/CNR neuroscience/cebra_codes/Cebra_for_all/cebra_codes'
os.chdir(main_path)
#main_path=r'/home/zlollo/CNR/Cebra_for_all'

#os.chdir(main_path)

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
    "max_iterations": "1000",
    "distance": "cosine",
    "conditional": "time_delta",
    "hybrid": "True",
    "time_offsets": "10"
    }



from hip_models_0 import run_hip_models
from fig_cebra import plot_cebra
from cr_db_sql import create_database
from cr_db_sql import save_fig
from cr_db_sql import save_manif
#from FIG2_mod import  Fig2_rat_hip
# Now you can call run_hip_models() in your script

def main(params):
    #create_database() 
    base_path=main_path
      

   # Fig2_rat_hip(dd, err_loss, mod_pred,base_path) 
    manif, labels = run_hip_models(base_path,params)
    #unique_name = "manif_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    #save_manif(manif, unique_name)
    #fig=plot_cebra(manif, labels)
    #save_fig(fig, fig_id="my_plot_id")
    plot_cebra(manif, labels)

    input("Press any key to continue..")

    #plot_cebra(manif, labels)

    return manif, labels
    
    #return  dd, err_loss, mod_pred
    
if __name__=="__main__":


    #dd, err_loss, mod_pred= 

    main(params)






#neur=hip_pos.neural.numpy()

#pippo1=dd['visualization']['hypothesis']
#pippo=dd['visualization']['discovery']
#