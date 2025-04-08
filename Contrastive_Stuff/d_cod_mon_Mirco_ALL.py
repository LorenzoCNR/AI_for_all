
import os
import sys
import math
from pathlib import Path
import argparse
import logging
import mimetypes
from statsmodels.tsa.seasonal import seasonal_decompose

### load path (use the function in module some functions)
'''
Assume a path structure:

main_folder
|___ data(folder)
|     |__project_data_folder1
                    |__ dati_cebra.jl (file dati)
      |__project_data_folder2
                    |__ dati_mirco.mat (file dati mat file, jle file etc)
|___ project_root(folder)
            |__d_cod_mon_Mirco.py
            |__some_functions.py
            |___EEG_ANN_pipeline(folder)
                    |__data (folder)
                    |__helpers (folder)
                    |__etc....
            |___output directory (folder)
            

'''
# need to declare:
# 1) PROJECT root directory
#  windows directories
i_dir='J:\\AI_PhD_Neuro_CNR\\Empirics\\GIT_stuff\\AI_for_all\\Contrastive_Stuff'

#  ubuntu directories
#i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'

os.chdir(i_dir)
os.getcwd()
from some_functions import *

# 2) NAME of folder containing data (input) directory. 
data_dir="data"
# (specific) project data folder
sub_data="Monkeys_Mirco"

# 3) PIPELINE folder name
pipe_path= "EEG-ANN-Pipeline"

# 4) OUTPUT folder: folder to store processed output
#    (if not existing is created)
out_dir="contrastive_output"

project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths( data_dir,sub_data,out_dir, pipe_path,change_dir=False)

# 
from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
from data.preprocessing import normalize_signals
from models import EncoderContrastiveWeights
from helpers.model_utils import plot_training_metrics, count_model_parameters, train_model
from helpers.visualization import plot_latent_trajectories_3d, plot_latents_3d
from helpers.distance_functions import *
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
torch.set_printoptions(sci_mode=False)

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


# BUILD the model encoder. (1: cnnd1d simil cebra with skip connections)
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

#### Build the model encoder (2: cnn2d)
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

###  DEFINE DISTANCES to compute the loss (In  a seprated module)
"""

- Minkowski Distance: A generalized distance metric that includes 
    both Euclidean (p=2) and Manhattan (p=1) distances as special cases. 

- Adaptive Gaussian Distance**:
- Direction Distance

- Circular Distance

"""

###  DEFINE DISTANCES to compute the loss (In   some_functions module)
"""

- Minkowski Distance: A generalized distance metric that includes 
    both Euclidean (p=2) and Manhattan (p=1) distances as special cases. 

- Adaptive Gaussian Distance**:
- Direction Distance

- Circular Distance

"""

### DECODING (to be done)
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
   
#### PARAMETERS ###
## sampling frequency (ms)    
fs = 1000
## Validation/test split
valid_split = 0.15
## window of each mini batch
ww = 10
# overlap
shift = 1
#
dropout = 0.5
## temperature
tau = 0.5
## network hidden layers channles
filters = 32
## output channels (latents)
latents = 3
# learning rate
l_rate = 0.0001
#
epochs = 100
#sigma_pos = 0.016
#sigma_time = 0.025
#
batch_size=1024
#
num_units = filters
#
normalize=True

# recall the input dir (declared upwards) and import data
input_dir = default_input_dir
# data format
d_format="mat"
# data_name
d_name='dati_mirco_18_03_joint'
data = load_data(input_dir, d_name,d_format)
print(type(data)) # must be a dictionary


#### define X ed Y
'''
Data Structure:
   - X (neural data) is a matrix T(ime)xch(annels)
   - y_... are the associated labels, which can be:
    Discrete or continuous.
    Either matrices or vectors with dimensions T×n, meaning:
        Labels exist for each time point.
        n is the number of label dimensions.


'''
#X=data['m1_active_neural'][:,[0,2]]
#X=data['m1_active_neural'][:,[0,2]]
X=data['joint_mix_neural']
y_dir=data['joint_mix_trial']
y_dir=y_dir.flatten()
#y_pos=data['mix_active_trial']
trial_id=data['joint_mix_trial_id']
trial_id=trial_id.flatten()
#n_trials_=127
len_y=len(y_dir)
change_idx = np.where(np.diff(trial_id) != 0)[0] + 1
change_idx
print('ciao')
# the c_t vector tells the starting and endiing points of every trial
c_t=np.concatenate([[0], change_idx,[len_y]], dtype=int)
## list of list of starting and ending points for trials

c_t_list=[]
c_t_list = [(c_t[i], c_t[i+1] - 1) for i in range(len(c_t) - 1)]
n_trials=len(c_t)-1
### check trial length (useful for graphics)
trial_len=np.diff(c_t)
trial_length=trial_len[0]

# direction_dict = {
#     1: 'nord',
#     2: 'nord_est',
#     3: 'est',
#     4: 'sud_est',
#     5: 'sud',
#     6: 'sud_ovest',
#     7: 'ovest',
#     8: 'nord_ovest'
# }

########################### RESAMPLING ########################################
#### freqeuncy of sampling(if you want to use data at a lower freq)
#sampling_freq=10

##### option to resample data a different frequency 
#### RESAMPLIGN DATA FUNCTION
for start_trial, end_trial in c_t_list:
    print(start_trial, end_trial)

step = 3
methods = {
    0: "sum",  
    1: "center"  
    #,2: "mean"
}

l_data=[X,y_dir] 
step=10
overlap=5
Normalize=True
### resampled data, new trials lengths, new trials intervals
resampled,r_trial_lengths,r_trial_indices =  f_resample(l_data,c_t_list, step,
                       overlap, methods, mode="overlapping",normalization=True)

r_trial=r_trial_indices[0]
start_points=[]
for sublist in r_trial:
    print(sublist[0])
    start_points.append(sublist[0])
r_last=r_trial[-1][1]
start_points.append(r_last+1)
c_t_resampled=np.array(start_points).flatten()

r_trial_indices[0][1][0]
unique_labels = np.unique(resampled[1])
### 6-3, 3-6
original_label_order = np.arange(1, 9)  # [1,2,3,4,5,6,7,8]

swap_dict = {3: 6, 6: 3}

# Apply mapping back to restore original labels (optional)
resampled_swapped = swap_labels(resampled[1], swap_dict)
if np.var(trial_len)==0:    
    constant_len=True
else:
    constant_len=False
print(constant_len)



####  Number of channels Either first (0) or second dimension (1)
chns = resampled[0].shape[1]
print(chns)
#y_pos=y_pos.flatten()



################################# check data ##################################
################################## PLOTS #####################################
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
n_tr=35
plt.title("Line graph") 
###♣ trial to watch
n_s_=10
## channel to watch
ch_=1

l_tt=c_t_resampled[n_tr]-c_t_resampled[n_tr-n_s_]
#l_tt=len(X)
tt=np.arange(1,l_tt+1)

###♦ trend line (Gaussian filter applied to every channel)
# # smoothing parameter
sigma = 10  
smoothed_series = np.zeros_like(X[tt, ch_])
for i in range(n_tr - n_s_, n_tr):
    start, end = c_t[i], c_t[i + 1]
    smoothed_series[start - c_t[n_tr - n_s_]:end - c_t[n_tr - n_s_]] = gaussian_filter1d(X[start:end, ch_], sigma=sigma)

# Compute upper envelope with hilbert transform
envelope_upper = np.abs(hilbert(X[tt, ch_]))


fig, axs = plt.subplots(2,2)
#plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],0],label='first channel', color ="red")
#span=
axs[0, 0].plot(tt, X[tt, ch_]**2, color="red", linewidth=1.5)
axs[0, 0].set_title('Squared Series', fontsize=14)
axs[0, 0].grid(True, linestyle='--', linewidth=0.5)
#plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],1],label='second channel', color ="blue") 

axs[0, 1].plot(tt, abs(X[tt, ch_]), color="black", linewidth=1.5)
axs[0, 1].set_title('Abs Series', fontsize=14)
axs[0, 1].grid(True, linestyle='--', linewidth=0.5)
#plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],2],label='third channel', color ="black") 

axs[1, 0].plot(tt, X[tt, ch_], color="blue", linewidth=1.2, label="Original series")
axs[1, 0].plot(tt, envelope_upper, color="green", linestyle="--", linewidth=1.5, label="Upper Envelope")  # Inviluppo superiore
axs[1, 0].set_title('Original Series', fontsize=14)
axs[1, 0].grid(True, linestyle='--', linewidth=0.5)
axs[1, 0].legend()#axs[1, 0].plot(tt, smoothed_series, color="orange", linewidth=2, label="Trend")  # Linea di tendenza


#plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],2],label='third channel', color ="black") 
axs[1, 1].plot(tt, X[tt, 2], color="gray", linewidth=1.5)
axs[1, 1].set_title('Third Channel', fontsize=14)
axs[1, 1].grid(True, linestyle='--', linewidth=0.5)

for ax in axs.flat:
    for i in range(n_tr - n_s_, n_tr):
        ax.axvline(x=c_t[i] - c_t[n_tr - n_s_], linestyle="--", color="black", linewidth=1.5)

plt.tight_layout()
plt.show()

#### Io osservando le serie farei una scomposizioen

##############################################################################



### RESAMPLING
#### freqeuncy of sampling(if you want to use data at a lower freq)
#sampling_freq=1
###copy data in order to re-sample
#data1=copy.deepcopy(data)   
##### option to resample data a different frequency 
#l_data=[data['active_target'], data['spikes_active'], data['pos_active'], 
#        data['num_trials']]
# #  data to be resampled 
# l_data=[ data['spikes_active'],data['active_target'], data['pos_active']]
# res_d=f_resample(l_data,10)
#X=res_d[0]
# y_dir=res_d[1]
# y_pos=res_d[2]

# #### PLOTS of active and passive movements and of neural data
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


######## CREATE TRIALS adding the behavioral info we want ########
'''
 trial_ids is a vector telling us the trial at time t
 c_t tells us starting (and ending) point of each trial 
'''
### SE USiamo dati resampled



   #◘ for i, r in enumerate(resampled):
  #  print(f"Dataset {i} resamplato: {r}")

r_trial_indices[0][1][0]
################################# OKKIO cambia c_t ############################
c_t=c_t_resampled
y_dir=resampled_swapped
X=resampled[0]
trials = [
    TrialEEG(
       X[ c_t[i]:c_t[i+1]].T,  # Segnali EEG
          # Labels
         # ,
          [
            ('position', y_dir[c_t[i]:c_t[i+1]].T),
           ( 'direction',y_dir[c_t[i]:c_t[i+1]].T        
       ) ],
          # Timepoints
        np.linspace(c_t[i] / fs, c_t[i+1] / fs, c_t[i+1] - c_t[i])  
    )
    for i in range(len(c_t) - 1)
]

y_dir_res=[]
for i in range(len(c_t)-1):
    print(i)
    y_dir_res.append(resampled[1][c_t[i]:c_t[i+1]-10].T)
y_dir_res_0 = np.concatenate(y_dir_res, axis=0) 

## check
explore_obj(trials[5])
dataset = DatasetEEG(trials)
print(dataset)
print(dataset.trials[1])
explore_obj(dataset.trials[1])
dataset.trials[0].eeg_signals.shape

### Function to generate embeddings after data processing 
# (move to some_functions after generalizing for the labels)
def generate_embeddings(model, dataset_pytorch, batch_size, device):
    model.eval()
    ## 
    z, labels_position, labels_direction = [], [],[]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    with torch.no_grad():
        for i in range(0, dataset_pytorch.num_trials, batch_size):
            x = dataset_pytorch.eeg_signals[i:i+batch_size].to(device)
            #l_pos = dataset_pytorch.labels['position'][i:i+batch_size,:]
            l_dir = dataset_pytorch.labels['direction'][i:i+batch_size]
            print(l_dir)
            f_x = model(x)

            z.append(f_x.cpu().numpy())
            #labels_position.append(l_pos.cpu().numpy())
            labels_direction.append(l_dir.cpu().numpy().reshape(-1))

    z = np.concatenate(z)
    #labels_position = np.concatenate(labels_position)
    labels_direction = np.concatenate(labels_direction)
 
    return z ,labels_direction
#, labels_position



############# FULL DATASET (No training/test/validation split) ################
'''
split dataset into windows (these are the elements going in the batches)
    ww: mini-sample dimension
    shift: window overlap (shift=1 means sampling 1-10, 2-11, 3-12...)
'''
### 
dataset_windows=dataset.create_windows(ww, shift)
print(dataset_windows)
# check
explore_obj(dataset_windows.trials[2500])

#### Convert to PyTorch datasets
# if u need to select some particular labels
#sel_lab='position' (optional argument for the next called function)

dataset_pytorch=DatasetEEGTorch(dataset_windows)
explore_obj(dataset_pytorch)

'''
# Define label distances...multi label must be a dictionary; single label can 
be  either a function or a dictionary with one element
 'direction': direction_distance
  'position':adaptive_gaussian_distance..

the following line includes both position and direction distancethat is both 
lables will be used during training
***labels_distance = LabelsDistance({'position': adaptive_gaussian_distance,'direction':direction_distance})

the following line only includes direction distance in the process
***labels_distance = LabelsDistance({'position': adaptive_gaussian_distance})

'''
labels_distance = LabelsDistance({'direction':direction_distance})
#labels_distance = LabelsDistance({'position': adaptive_gaussian_distance})

explore_obj(labels_distance)
## deliver to cuda devices (if any available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_pytorch.to_device(device)
device
#batch_size=1024

dataloader_ = DataLoader(dataset_pytorch, batch_size=batch_size, shuffle=False)
## explore the dataloader
for batch in dataloader_:
    #►pippo.append(batch)
    print(type(batch[1])) 
    print(batch[0].shape)       
    break 

#################################################################################
# Build and train the model
# update the weights according the chosen labels (same value, same weight)
# just one value for label

from data import LabelsDistance, TrialEEG, DatasetEEG, DatasetEEGTorch
from data.preprocessing import normalize_signals
from models import EncoderContrastiveWeights
filters=64
num_units=filters
epochs=1000
tau=2
model = EncoderContrastiveWeights(
    ### define the network (recall the parameters defined from line 209)
    layers=build_model(filters, dropout, latents, ww, chns, normalize=normalize),
    ### choose between infoNCE, supervised contrastive, weeighted contrastive loss
    #loss_type="weighted_contrastive",
    # 
    labels_distance=labels_distance,
    #labels_distance=None,
    ## If we use the labels, choose some weights
    ## if we have more label and want a different weight for every label
    ## just put a weight for each label; if we have one label or want just the 
    ## same weight for each label one value is ok
    labels_weights=[1],
    #labels_weights=None,
    temperature=tau,
    train_temperature=False,
    ### time contrastive loss asks either offset or window
    #positive_offset=5,  # Offset per i positivi (se usi offset)
    #positive_window=0 
    
    )

model.to(device)

print("Model architecture:", model)
print(f"Model has {count_model_parameters(model)} parameters.")
#print(f"subject is {name}")
print(f"{device}")

# Train and evaluate the model
optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

# train on a batch (just to check if everything is ok)
batch = next(iter(dataloader_))
loss_dict = model.process_batch(batch, optimizer)

### model training
metrics = train_model(model, optimizer, dataloader_, epochs=epochs)
plot_training_metrics(metrics)


from some_functions import plot_embs_continuous, explore_obj,plot_direction_averaged_embedding
## GENERATE EMBEDDINGS and recover the label info 
#, l_pos_0 
z_0, l_dir_0= generate_embeddings(model, dataset_pytorch, batch_size, device)

### generate 3d plot
### DATI RESAMPLED
trial_length_r=r_trial_lengths[0][0]
#trial_length=trial_len[0]
constant_len=True
###define the trajectories. First monkey is 1-8, second is 9-16
# first
n_traj=np.arange(8)
# second
#n_traj=np.arange(8)+9
#

c_s="maroon"
title_=f"C_L_platform_cond3_shift_5_lr.html"

output_folder = default_output_dir
results_list=[]
plot_direction_averaged_embedding(
            z_0,
            y_dir_res_0,
            original_label_order,
            c_s,
            output_folder,
            title_,
            trial_length_r,
            constant_length=True,
            ww=10,
            label_swap_info=swap_dict
        ) 



z_=z_0
l_dir_=l_dir_0
ww=10


for idx, original_label in enumerate(original_label_order):
    print(original_label)
    mask=(l_dir_==original_label)
    count=mask.sum()
    count
    trial_avg = z_[mask].reshape(-1, trial_length_r - ww, 3).mean(axis=0)
    
print(f"original_label: {original_label}")
print(f"count (mask.sum()): {count}")
print(f"z_[mask].shape: {z_[mask].shape}")
print(f"trial_length - ww: {trial_length - ww}")
    
    
'''
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
'''

