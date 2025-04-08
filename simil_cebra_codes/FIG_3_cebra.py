# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:14:59 2025

@author: loren
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
print(sklearn.__version__)
fig_dir=r'J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\cebra-figures\data\Figure3.h5'

data_f = pd.read_hdf(fig_dir, key = "data")
true = data_f["trajectory"]["true"]
pred = data_f["trajectory"]["prediction"]
true = data_f["trajectory"]["true"]
pred = data_f["trajectory"]["prediction"]
direction_acc = data_f["condition_decoding"]["direction"]
position_acc = data_f["condition_decoding"]["position"]
ap_acc = data_f["condition_decoding"]["ap"]
overview = data_f["overview"]


#### FIGURA 3B Comparison of embeddings of active trials generated with
# CEBRA-Behavior, CEBRA-Time, conv-pi-VAE variants, tSNE, and UMAP. 
# The embeddings of trials (n=364) of each direction are post-hoc averaged.
fig = plt.figure(figsize=(30, 5))
plt.subplots_adjust(wspace=0, hspace=0)
emissions_list = [
    overview["cebra-behavior"],
    overview["pivae_w"],
    overview["cebra-time"],
    overview["pivae_wo"],
    overview["autolfads"],
    overview["tsne"],
    overview["umap"],
]

labels = overview["label"]
trials = emissions_list[0].reshape(-1, 600, 4)
trials_labels = labels.reshape(-1, 600)[:, 1]
mean_trials = []
idx1, idx2 = (2, 0)
mean_trial = trials.mean(axis=0)
for i in range(8):
            mean_trial = trials[trials_labels == i].mean(axis=0)
            mean_trials.append(mean_trial)
for trial, label in zip(mean_trials, np.arange(8)):
            plt.plot(
                trial[:, idx1], trial[:, idx2], color=plt.cm.hsv(1 / 8 * label),
                #linewidth = lw
            )
    
for trial, label in pippo:
    
    print(trial.shape(), label.shape())
    
trial.shape
label.shape

## #alternativa
#Plot delle traiettorie nel tempo
# Se l'obiettivo è vedere la dinamica della posizione, potresti visualizzare le traiettorie latenti nel tempo:
### Fig 3 e alternativa
features_pos = data_f["behavior_time"]["behavior"]
    

plt.plot(features_pos[:, 0], features_pos[:, 1], color='gray', alpha=0.5)
plt.scatter(features_pos[:, 0], features_pos[:, 1], c=labels_pos[:, 0], cmap="jet", s=1)
plt.colorbar(label="X Position")
plt.title("Latent space trajectory colored by X position")
plt.show()


