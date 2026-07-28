# -*- coding: utf-8 -*-
"""
Created on Tue May  6 19:42:51 2025

@author: loren
"""
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA

monkey_pos = cebra.datasets.init('area2-bump-pos-active')
monkey_target = cebra.datasets.init('area2-bump-target-active')

cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.0001,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=10000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)


ppos=monkey_pos.continuous_index.numpy()

ttar=np.array(monkey_target.discrete_index.numpy(),dtype=np.float16)
ttar=ttar.reshape(len(ttar),1)
labels_ =np.concatenate((ppos,ttar),axis=1)


cebra_pos_model.fit(monkey_pos.neural, labels_)
cebra_target = cebra_pos_model.transform(monkey_pos.neural)

%matplotlib notebook
fig = plt.figure(figsize=(4, 2), dpi=300)
plt.suptitle('CEBRA-behavior trained with target label',
             fontsize=5)
ax = plt.subplot(121, projection = '3d')
ax.set_title('All trials embedding', fontsize=5, y=-0.1)
x = ax.scatter(cebra_target[:, 0],
               cebra_target[:, 1],
               cebra_target[:, 2],
               c=monkey_target.discrete_index,
               cmap=plt.cm.hsv,
               s=0.01)
ax.axis('off')

ax = plt.subplot(122,projection = '3d')
ax.set_title('direction-averaged embedding', fontsize=5, y=-0.1)
for i in range(8):
    direction_trial = (monkey_target.discrete_index == i)
    trial_avg = cebra_target[direction_trial, :].reshape(-1, 600,
                                                         3).mean(axis=0)
    trial_avg_normed = trial_avg/np.linalg.norm(trial_avg, axis=1)[:,None]
    ax.scatter(trial_avg_normed[:, 0],
               trial_avg_normed[:, 1],
               trial_avg_normed[:, 2],
               color=plt.cm.hsv(1 / 8 * i),
               s=0.01)
ax.axis('off')
plt.show()

