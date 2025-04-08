
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
filters = 16
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
d_name='dati_mirco'
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
X=data['m1_active_neural']

y_dir=data['m1_active_trial']
y_dir=y_dir.flatten()
y_pos=data['m1_active_pos']
### Data augmentation and -re-scaling
X=pd.DataFrame(X)
X.iloc[:,    1]
result = seasonal_decompose(X.iloc[:,1], model='additive')
print(result.trend)
print(result.seasonal)
print(result.resid)
print(result.observed)



X.mean()
X.std()
### rescale
X=X*100
X_abs=np.abs(X)
X_vol=(X)**2
#X_bpf=
from statsmodels.tsa.seasonal import STL
series = pd.Series(X[:,1])  # Converti in Pandas
stl = STL(series,period=1000, seasonal=101, robust=True)  # seasonal deve essere un numero alto per dati ad alta frequenza
res = stl.fit()
detrended_data = res.trend  # 


diff_X=first_difference(X,c_t)
diff_X_=np.concatenate(diff_X)
X=diff_X_
#=np.concatenate([X,diff_X_],axis=1)
####  Number of channels Either first (0) or second dimension (1)
chns = X.shape[1]
print(chns)
#y_pos=y_pos.flatten()


# IDENTIFY Trials (info should  be in the data dictionary or should be derived somehow
# i.e. every change in y_dir value - direction label - 
# tipo dove y_dir cambia...with the caution that it might not change from one
# trial to another).

trial_id=data['m1_active_trial_id']
trial_id=trial_id.flatten()
#n_trials_=127
len_y=len(y_dir)
change_idx = np.where(np.diff(trial_id) != 0)[0] + 1
change_idx
print('ciao')

# the c_t vector tells the starting and endiing points of every trial
c_t=np.concatenate([[0], change_idx,[len_y]], dtype=int)
n_trials=len(c_t)-1
### check trial length (useful for graphics)
trial_len=np.diff(c_t)

# zero variance = fix (constant) length (this info will be used for plots)
if np.var(trial_len)==0:    
    constant_len=True
else:
    constant_len=False
print(constant_len)





################################# check data ##################################

n_tr=35
plt.title("Line graph") 

l_tt=c_t[n_tr]-c_t[n_tr-1]
tt=np.arange(1,l_tt+1)
plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],0],label='first channel', color ="red") 
plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],1],label='second channel', color ="blue") 
plt.plot(tt,X[c_t[n_tr-1]:c_t[n_tr],2],label='third channel', color ="black") 

plt.xlabel('Time')
plt.ylabel('freq?')
plt.legend()
plt.show()


##############################################################################

#### Io osservando le serie farei una scomposizioen


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

trials = [
    TrialEEG(
       X[ c_t[i]:c_t[i+1]].T,  # Segnali EEG
          # Labels
          [(
             'position', y_pos[c_t[i]:c_t[i+1]].T),
           ( 'direction',y_dir[c_t[i]:c_t[i+1]].T,          
       ) ],
          # Timepoints
        np.linspace(c_t[i] / fs, c_t[i+1] / fs, c_t[i+1] - c_t[i])  
    )
    for i in range(len(c_t) - 1)
]

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
            l_pos = dataset_pytorch.labels['position'][i:i+batch_size,:]
            l_dir = dataset_pytorch.labels['direction'][i:i+batch_size]
            print(l_dir)
            f_x = model(x)

            z.append(f_x.cpu().numpy())
            labels_position.append(l_pos.cpu().numpy())
            labels_direction.append(l_dir.cpu().numpy().reshape(-1))

    z = np.concatenate(z)
    labels_position = np.concatenate(labels_position)
    labels_direction = np.concatenate(labels_direction)
 
    return z ,labels_direction, labels_position



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
explore_obj(dataset_windows.trials[46464])

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
#labels_distance = LabelsDistance({'direction':direction_distance})
labels_distance = LabelsDistance({'position': adaptive_gaussian_distance})

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
filters=32
num_units=filters
epochs=100
tau=0.2
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
    labels_weights=[0.5],
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


from some_functions import plot_embs_continuous, explore_obj
## GENERATE EMBEDDINGS and recover the label info 
z_0, l_dir_0, l_pos_0 = generate_embeddings(model, dataset_pytorch, batch_size, device)

### generate 3d plot
trial_length=trial_len[0]
constant_len
###define the trajectories. First monkey is 1-8, second is 9-16
# first
n_traj=np.arange(8)+1
# second
#n_traj=np.arange(8)+9
#

c_s="maroon"
output_folder = default_output_dir
plot_direction_averaged_embedding(z_0, l_dir_0, n_traj,c_s,output_folder, 
                                      trial_len, constant_length=True, 
                                      ww=10)





    
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

