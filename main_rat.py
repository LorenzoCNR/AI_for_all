#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 01:47:54 2023

@author: zlollo
"""

import os
##### cambiare eventualmente
main_path=r'/home/zlollo/CNR/Cebra_for_all'
os.chdir(main_path)

os.getcwd()
#from pathlib import Path

# IMAGES_PATH = Path() / "images" 
# IMAGES_PATH.mkdir(parents=True, exist_ok=True)

# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)




from hip_models import run_hip_models
from FIG2_mod import  Fig2_rat_hip
# Now you can call run_hip_models() in your script

def main():
    base_path=main_path
    dd, err_loss, mod_pred = run_hip_models(base_path)     
    Fig2_rat_hip(dd, err_loss, mod_pred,base_path) 
    
    return  dd, err_loss, mod_pred
    
if __name__=="__main__":
    #dd, err_loss, mod_pred= 
     main()



#neur=hip_pos.neural.numpy()

#pippo1=dd['visualization']['hypothesis']
#pippo=dd['visualization']['discovery']
#