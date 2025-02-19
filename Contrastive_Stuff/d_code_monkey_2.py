#!/usr/bin/env python
# coding: utf-8



# In[215]:
# windows
# input_dir=r'F:\........'
# main_root=r'F:\....'

#### Ubuntu
#main_root = r"/media/zlollo/........."
#input_dir = r"/media/zlollo/........."

#### Quindi fittare il modello sui dati neurali e posizione
##### sui dati neurali e il target
#### solo time

'''
Se ho capito bene abbiamo ttto diviso in trials

- tLFP tempo dei local field potential
- LFP local field potential; somma attività neurale di blocchi di 
  neuroni(2 scimmie)

JXEXY immaggino siano i dati (primi 4 scimmia 1 ....gli altri 4 scimmia 2 )
timeJXYEXY: credo sia il tempo
dt è l'intervallo

iLFP_K  
iLFP_S questo è il precdente potrebbero essere consegenza di un
processing sui dati

ARp (autoregulation parameter)
rsfactor: rescaling or normaliing factor
AlignEvent
Chamber 
session__name va da sè

RT_S reaction time of subject in some task
RT_K 
ET time marker for specific experimental event
timeET timing et events



'''


import os
import sys
import math
from pathlib import Path
import mimetypes
torch.set_printoptions(sci_mode=False)
# 
from some_functions import plot_embs, explore_obj, load_data
from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
from data.preprocessing import normalize_signals
from models import EncoderContrastiveWeights
from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
from layers.custom_layers import _Skip, Squeeze, _Norm, _MeanAndConv
# 
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
import random
from torch import nn
import torch
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import joblib as jl
import seaborn as sns# 
# 
from cebra import CEBRA
import cebra
from some_functions import *
# 
import matplotlib

# Random Seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Config GPU
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# backend or inline plots
# %matplotlib inline
# matplotlib.use('Agg')
# matplotlib.use('TkAgg')
# matplotlib.use('QtAgg')  
# from data.eeg_dataset import *
# plt.ion()
# plt.show()
# plt.pause(10)  

# Load data from the specified path
#default_input_dir='/media/lorenzo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/data/monkey_reaching_preload_smth_40'
# Build the model encoder.
def build_model(filters, dropout, latents, num_timepoints, chns, num_units=None, groups=1,normalize=True):
    """
    Build a cnn1d model with:
    - chns: Input channels.
    - filters: convolutional layer(s) filters.
    - latents: outpuit dimension (latent space).
    - num_timepoints: window dimension (test the optimal one).
    - dropout:  dropout.
    - num_units: optional intermediate filters.
    """
    if num_units is None:
        num_units = filters 

    layers = [
        Squeeze(),
        nn.Conv1d(chns, filters, kernel_size=2),
        nn.GELU(),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        _Skip(nn.Conv1d(filters, filters, kernel_size=3), nn.GELU()),
        nn.Conv1d(filters, latents, kernel_size=3),
    ]

    if normalize:
        layers.append(_Norm())  #

    layers.extend([
        nn.Flatten(),  # 
        #nn.Dropout(dropout),  # 
    ])

    return nn.Sequential(*layers)

#### Build the model encoder. 
# 
'''
def build_model(filters, dropout, latents, num_timepoints, chns):
    
    return nn.Sequential(
        nn.Conv2d(1, filters, kernel_size=(1, num_timepoints)),
        nn.BatchNorm2d(filters),
        nn.Conv2d(filters, filters, kernel_size=(chns, 1), groups=filters),
        nn.BatchNorm2d(filters),
        nn.Dropout(dropout),
        nn.Flatten(),
        nn.Linear(filters, filters),
        nn.SELU(),
        nn.Dropout(dropout),
        nn.Linear(filters, latents)
    )

'''

###  DEFINE DISTANCES 

#def position_distance(l1, l2, sigma_pos):
#    return torch.exp(- torch.sum((l1 - l2)**2, dim=-1) / (2 * sigma_pos**2))


def adaptive_gaussian_distance(x1:torch.tensor, x2:torch.tensor)->torch.Tensor:
    """
    Calcola la distanza normalizzata usando la varianza empirica dei dati.

    Args:
        x1, x2 (torch.Tensor): Tensor di forma (batch_size, D).

    Returns:
        torch.Tensor: Matrice (batch_size, batch_size) con le distanze normalizzate.
    """
    sigma = torch.std(x1, dim=0, unbiased=True, keepdim=True)  # Deviazione standard per feature  # Calcola la deviazione standard empirica
    sigma=sigma.mean()
    dists = torch.cdist(x1, x2, p=2)
    return torch.exp(- (dists**2) / (2 * sigma**2))  # Normalizzazione Gaussiana



# ### generalization   
# def minkowski_distance(l1, l2, p=2, sigma=1.0):
#     return torch.exp(- (torch.sum(torch.abs(l1 - l2)**p, dim=-1)**(1/p)) / (2 * sigma**2))    
# #### ...normalized according to variance 
# def minkowski_normalized_distance(coordinates, p=2):
#     # Calcolo delle deviazioni standard per normalizzazione
#     std_dev = torch.std(coordinates, dim=0)
    

#     # Normalize coordinates
#     coordinates_normalized = coordinates / std_dev
    
#     # Compute normalized minkowski distance
#     return torch.cdist(coordinates_normalized, coordinates_normalized, p=p)

# def time_distance(l1, l2, sigma_time):
#     return torch.exp(-(l1 - l2)**2 / (2 * sigma_time**2))



def direction_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    return l1[:, None] == l2[None, :].int() 



#### Generate embeddings according to the run model 
def generate_embeddings(model, dataset_pytorch, batch_size, device):
    model.eval()
    z, labels_position, labels_direction = [], [],[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    with torch.no_grad():
        for i in range(0, dataset_pytorch.num_trials, batch_size):
            x = dataset_pytorch.eeg_signals[i:i+batch_size].to(device)
            l_pos = dataset_pytorch.labels[i:i+batch_size,:]
            l_dir = dataset_pytorch.labels[i:i+batch_size]
            #print(l_dir)
            f_x = model(x)

            z.append(f_x.cpu().numpy())
            labels_position.append(l_pos.cpu().numpy())
            labels_direction.append(l_dir.cpu().numpy().reshape(-1))

    z = np.concatenate(z)
    labels_position = np.concatenate(labels_position)
    #labels_direction_1 = np.concatenate(labels_direction)
    #labels_direction_2 = 1 - labels_direction_1
    #labels_ = np.stack(labels_direction_1, axis=1)
    #labels_ = np.stack((labels_position, labels_direction_1,
     #                  labels_direction_2), axis=1)
    #return z, labels_
    return z, labels_position

z_,labels_ = generate_embeddings(model, dataset_pytorch, batch_size, device)


# ###decoder
# def decoding_knn(embedding_train, embedding_test, label_train, label_test,metric, n_n):
#     #metric = 'cosine'
#     #n_n = 25
#     pos_decoder = KNeighborsRegressor(n_neighbors=n_n, metric=metric)
#     dir_decoder = KNeighborsClassifier(n_neighbors=n_n, metric=metric)

#     pos_decoder.fit(embedding_train, label_train[:, 0])
#     dir_decoder.fit(embedding_train, label_train[:, 1])

#     pos_pred = pos_decoder.predict(embedding_test)
#     dir_pred = dir_decoder.predict(embedding_test)

#     prediction = np.stack([pos_pred, dir_pred], axis=1)

#     test_score = sklearn.metrics.r2_score(
#     label_test[:, :2], prediction, multioutput='variance_weighted')
#     max_label = np.max(label_test[:, 0])
#     print(f'Max value of label_test[:, 0] is: {max_label}')
    
#     pos_test_err_perc = np.median(abs(prediction[:, 0] - label_test[:, 0]) / max_label * 100)
    
#     pos_test_err = np.median(abs(prediction[:, 0] - label_test[:, 0]))
    
#     pos_test_score = sklearn.metrics.r2_score(
#         label_test[:, 0], prediction[:, 0])

#     return test_score, pos_test_err,pos_test_err_perc, pos_test_score


#def run_d_code(input_dir, output_dir, name, filters, tau, epochs, dropout, latents, ww, sigma_pos, sigma_time, train_split, valid_split, l_rate, batch_size, fs, shift,normalize, neighbors):
    # load data
    
    
fs = 1000
valid_split = 0.15
ww = 10
shift = 1
dropout = 0.5
tau = 0.02
filters = 32
latents = 3
l_rate = 0.0001
epochs = 50
sigma_pos = 0.016
sigma_time = 0.025
batch_size=1024
num_units = filters
normalize=True
project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths()
#from d_code_monkey import run_d_code
name='SK009_Frontal_AllLFP.mat'
input_dir = default_input_dir
#path =default_input_dir / f"{name}.jl"
data = load_data(input_dir, name)
from some_functions import *   
###copy data in order to re-sample
data1=copy.deepcopy(data)   
##### dataset da resamplare 
l_data=[data['active_target'], data['spikes_active'], data['pos_active'], 
        data['num_trials']]

 #### definiamo X ed Y
X=data['spikes_active']
y_dir=data['active_target']
y_pos=data['pos_active']
n_trials=data['num_trials']

# #  data to be resampled 
# l_data=[ data['spikes_active'],data['active_target'], data['pos_active']]
# res_d=f_resample(l_data,10)

#X=res_d[0]
# y_dir=res_d[1]
# y_pos=res_d[2]

chns = X.shape[1]
# # print(data['spikes'].shape)
# # print(data['position'].shape)
# print(chns)
# #### Plots of active and passive movements and of neural  data
# active_target=y_dir
# active= y_pos
# fig = plt.figure(figsize=(8, 4))
# ax1 = plt.subplot(121)
# ax1.scatter(active[:, 0], active[:, 1], color=plt.cm.hsv(1 / 8 * active_target), s=1)
# ax1.axis("off")

# passive_target=data['passive_target']
# passive= data['pos_passive']
# ax2 = plt.subplot(122)
# ax2.scatter(passive[:, 0], passive[:, 1], color=plt.cm.hsv(1 / 8 * passive_target), s=1)
# ax2.axis("off")

# #### attività neurale###
# ### da capire cosa ci sta nei dati
# ### l'attività neurale, sono registrazioni LFP che riflettono l'attività
# # sinaptica locale. 
 
# fig = plt.figure(figsize=(15, 5))
# ephys = data["ephys"]
# ax = plt.subplot(111)
# ax.imshow(ephys[:6000].T, aspect="auto", cmap="gray_r", vmax=1, vmin=0)
# plt.ylabel("Neurons", fontsize=20)
# plt.xlabel("Time (s)", fontsize=20)
# plt.xticks([0, 200, 400, 600], ["0", "200", "400", "600"], fontsize=20)
# plt.yticks(fontsize=20)
# plt.yticks([25, 50], ["0", "50"])

# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)


######## create trials adding the behavioral info we want ########

### trial_ids is a vector telling us the trial at time t
### c_t is tll us when each trial starts (and ends)
trial_ids=y_dir
c_t=np.linspace(0,len(trial_ids), data['num_trials']+1, dtype= int)


''' VECCHIA ROBA
#### spikes ed entrambe (2) label, info comportamentali,

spikes_and_label = np.concatenate((data['spikes_active'], data['pos_active'], data['active_target']), axis=1)    
trials = [TrialEEG(spikes_and_label[c_t[i]:c_t[i+1]].T, 0, np.linspace(c_t[i] / fs, c_t[i+1] / fs, c_t[i+1] - c_t[i]))
          for i in range(len(c_t) - 1)]

'''

trials = [
    TrialEEG(
       X[ c_t[i]:c_t[i+1]].T,  # Segnali EEG
          # etichette
          [(
             'position', y_pos[c_t[i]:c_t[i+1]].T),
           ( 'direction',y_dir[c_t[i]:c_t[i+1]].T,          
       ) ],
        np.linspace(c_t[i] / fs, c_t[i+1] / fs, c_t[i+1] - c_t[i])  # Timepoints
    )
    for i in range(len(c_t) - 1)
]


from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
from data.preprocessing import normalize_signals
from models import EncoderContrastiveWeights
from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
# 
explore_obj(trials[192])
dataset = DatasetEEG(trials)
print(dataset)
print(dataset.trials[1])

############# FULL DATASET
#### split dataset into windows (these are the elements going in the batches)
dataset_windows=dataset.create_windows(ww, shift)
print(dataset_windows)
explore_obj(dataset_windows.trials[46464])
#### Convert to PyTorch datasets
# CHECK se funziona la selezione delle labels
#sel_lab='direction'
dataset_pytorch=DatasetEEGTorch(dataset_windows)
#explore_obj(dataset_pytorch[5].labels)
 # Define label distances...multi label must be a dictionary; single label can be
# either a function or a dictionary with one element
# 'direction': direction_distance
labels_distance = LabelsDistance({'direction': direction_distance, 'position':adaptive_gaussian_distance})
explore_obj(labels_distance)


## deliver to cuda devices (if any available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_pytorch.to_device(device)
dataloader_ = DataLoader(dataset_pytorch, batch_size=512, shuffle=True)
### explore the dataloader
pippo= []
for batch in dataloader_:
    pippo.append(batch)
    print(type(batch)) 
    print(batch)       
    break 
pippo[0]
#################################################################################

# Build and train the model
# update the weights according the chosen labels
model = EncoderContrastiveWeights(
    layers=build_model(filters, dropout, latents, ww, chns, normalize=True),
    labels_distance=labels_distance,
    labels_weights=[1,1],
    temperature=tau,
    train_temperature=True
)
model.to(device)

print("Model architecture:", model)
print(f"Model has {count_model_parameters(model)} parameters.")
print(f"subject is {name}")
print(f"{device}")


# Train and evaluate the model
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
   # print(f"psi shape: {psi.shape}")
#print(f"weights_log shape: {weights_log.shape}")
metrics = train_model(model, optimizer, dataloader_, epochs=epochs)


from some_functions import plot_embs_continuous, explore_obj
z_,labels_ = generate_embeddings(model, dataset_pytorch, batch_size, device)
   
title='plot_cont'
plot_embs_continuous(z_, labels_, title)

    

#### split data in training and validation...same steps
#################################################################################
dataset_training, dataset_validation = dataset.split_dataset(
    validation_size=valid_split)
explore_obj(dataset_training)
print(dataset_training)
print(dataset_training.trials[0].eeg_signals.shape)

# Create windowed datasets
dataset_windows_training= dataset_training.create_windows(ww, shift)
dataset_windows_validation = dataset_validation.create_windows(ww, shift)
print(dataset_windows_training)
explore_obj(dataset_windows_training)

# convert to torch
dataset_training_pytorch=DatasetEEGTorch(dataset_windows_training, sel_lab)
dataset_validation_pytorch = DatasetEEGTorch(dataset_windows_validation, sel_lab)
explore_obj(dataset_training_pytorch)

## deliver to cuda devices (if any available)

dataset_training_pytorch.to_device(device)
dataset_validation_pytorch.to_device(device)
#torch.cuda.init()

## create dataloader

dataloader_training = DataLoader(dataset_training_pytorch, batch_size=512, 
                                 shuffle=True)
    
dataloader_validation = DataLoader( dataset_validation_pytorch, batch_size=512,
                                   shuffle=False)
explore_obj(dataloader_training)

# Build and train the model
model = EncoderContrastiveWeights(
    layers=build_model(filters, dropout, latents, ww, chns, normalize=True),
    labels_distance=labels_distance,
    labels_weights=[1],
    temperature=tau,
    train_temperature=True
)
model.to(device)
print("Training dataset size:", len(dataset_training.trials))
print("Validation dataset size:", len(dataset_validation.trials))
print("Model architecture:", model)
print(f"Model has {count_model_parameters(model)} parameters.")
print(f"subject is {name}")
print(f"{device}")


# Train and evaluate the model
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
   # print(f"psi shape: {psi.shape}")
#print(f"weights_log shape: {weights_log.shape}")
metrics = train_model(model, optimizer, dataloader_, epochs=epochs)

# Save metrics and model
#torch.save(model.state_dict(), f"{output_dir}/{name}_model.pth")
#np.save(f"{output_dir}/{name}_metrics.npy", metrics)
plot_training_metrics(metrics)

 # Evaluate the model and plot latent spaces
#model Evali
# Generate embeddings for training data

from some_functions import plot_embs_discrete, explore_obj
z_train, labels_train = generate_embeddings(model, dataset_pytorch, batch_size, device)
   
# Generate embeddings for validation data
z_val, labels_val = generate_embeddings(model, dataset_validation_pytorch, batch_size, device)
   
# print(z.head())
#plt.figure()
title_= f"{name} - temp={tau}, epochs={epochs}, filters={filters}"
### ratio is the orginal  trial length divided by the sampling freq
ratio=int(600)
plot_embs_discrete(z_train, labels_train,title_,ratio,ww)
plt.show()
#input("Input Return to close graph..")
#n_n=25
metrics='cosine'
posdir_decode_CL = decoding_knn(z_train, z_val, labels_train, labels_val,metrics, neighbors)

return z_train, z_val ,labels_train, labels_val,posdir_decode_CL

   

   #  if __name__ == "__main__":
#    # #
 #        input_dir = r"F:\CNR_neuroscience\Consistency_across\Codice Davide"
  #       output_dir = r"F:\CNR_neuroscience\Consistency_across\Codice Davide"
#         name='rat_name'
#         run_d_code(input_dir, output_dir, name, filters, tau, epochs, dropout, latents, ww, sigma_pos, sigma_time, train_split, valid_split, l_rate, Batch_size):


