#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 13:19:21 2025

@author: zlollo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
trial_length=trial_len  
constant_length=constant_len
tr_se=[]


def plot_direction_averaged_embedding(z_, l_dir, n_traj, trial_length=None, 
                                      constant_length=True, ww=10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('direction-averaged embedding', fontsize=10, y=-0.1)
    
    # create unit sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.05, edgecolor='k', linewidth=0.2)
   
    for i in n_traj:
        
        direction_trial = (l_dir_0 == i)
        trial_data = z_0[direction_trial, :]
        #variable_length=True
       
        #direction_trial=(l_dir_0== 2)
        #trial_data = z_0[direction_trial, :]
        
        
        
        if constant_length:
            tr_length=trial_len[0]
            trial_avg = trial_data.reshape(-1, tr_length-ww, 3).mean(axis=0)
            #trial_avg -= trial_avg
            trial_avg_normed = trial_avg / np.linalg.norm(trial_avg, axis=1, keepdims=True)
            #trial_avg_normed = trial_avg.mean(axis=0)
            #trial_len=trial_length-ww
            ax.plot(trial_avg_normed[:, 0],
                    trial_avg_normed[:, 1], 
                    trial_avg_normed[:, 2], 
                    color=plt.cm.hsv( i/len(n_traj)),
                    linewidth=2, alpha=0.6, label=f"Dir {i}")
            ax.scatter(trial_avg_normed[0, 0], 
                       trial_avg_normed[0, 1],
                       trial_avg_normed[0, 2],
                       color='red', marker='o', s=20,
                       label="Start" if i == 0 else "")
            ax.scatter(trial_avg_normed[-1, 0], 
                       trial_avg_normed[-1, 1], 
                       trial_avg_normed[-1, 2], 
                       color='blue', marker='x',
                       s=20, label="End" if i == 0 else "")
     
        else:
            # length for each trial
            trial_length=trial_len
            tr_length = [length - ww for length in trial_length]

            # split data in segments of givne (variable) lengths
            trial_segments = np.split(trial_data, np.cumsum(tr_length[:-1]))
            tr_se.append(trial_segments)
        
            # check the ratio is ok
            assert len(trial_segments) == len(trial_len), "Errore: numero di segmenti non corrisponde alle lunghezze!" 
        
            # Normalizzazione e media per ogni segmento
            trial_avg = [seg / np.linalg.norm(seg, axis=1, keepdims=True) for seg in trial_segments]  # Normalizzazione
            trial_avg_normed = [seg.mean(axis=0) for seg in trial_avg]  # Media di ogni trial 
               
            
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=2, alpha=0.6, label=f"Dir {i}")
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], color='red', marker='o', s=20, label="Start" if i == 0 else "")
            ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], color='blue', marker='x', s=20, label="End" if i == 0 else "")
    
    
    ax.legend()
    plt.show()

n_traj=np.arange(8)+1


plot_direction_averaged_embedding(z_0, l_dir_0, n_traj, trial_len, 
                                      constant_length=constant_len, ww=10)