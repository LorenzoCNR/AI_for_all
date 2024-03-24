# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!pip install --pre 'cebra[dev,demos]'
#!ls
#!pip uninstall -y dreimac
#!pip uninstall -y cebra
#!pip uninstall -y ripser

import os
#os.getcwd()
################################## CAMBIARE ########################################
#main_folder=r'/home/zlollo/CNR/git_out_cebra'
#main_folder='/media/zlollo/STRILA/CNR neuroscience/'
#main_folder=r'
#os.chdir(main_folder)
import sys

#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')
#base_dir=r'/media/zlollo/STRILA/CNR neuroscience/cebra_codes'
# base_dir=r'C:\Users\zlollo2\Desktop\Strila_20_03_24\CNR neuroscience\cebra_codes'
# import time
# os.chdir(base_dir)
# output_directory =base_dir

#!pip install ripser
#import ripser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
#from dataset import SingleRatDataset  # Assumendo che il codice sia in 'dataset.py'
from matplotlib.collections import LineCollection
import pandas as pd
import matplotlib.lines as mlines
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
from scipy.io import loadmat
from cebra.datasets.hippocampus import *
import sklearn.metrics
import scipy.io
import joblib

from scipy.io import savemat
from scipy.io import loadmat
import gc
import torch
#import tensorflow as tf

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics

# Check # argomenti corretto
if len(sys.argv) != 3:
    print("Usage: python script.py <input_directory> <output_directory>")
    sys.exit(1)

input_directory = sys.argv[1]
output_directory = sys.argv[2]
# ############## DATI RATTO #################################
data_mat= loadmat('rat_n.mat')
data_=data_mat['rat_n']
label_mat=loadmat('rat_b.mat')
label_=label_mat['rat_b']
##################### PArametri Generici ###########################àssss
###############à###### Load model hyperparameters #############################


try:
    
    m_params=loadmat('params.mat')
    model_params= m_params['params']
   #data_param = loadmat('params.mat', squeeze_me=True, struct_as_record=False)
    
    #main_fold_array= gen_params['main_fold'][0]
    #main_fold= main_fold[0]
    #main_fold = ''.join(main_fold_array) 
    
    mod_type= model_params['model_type'][0].item() 
    mod_type= mod_type[0]

    
    #data_param=loadmat('model_params.mat')
        
    #model_params = data_param['model_params']
        
    mod_arch= model_params['mod_arch'][0].item() 
    mod_arch= mod_arch[0]
    
    dist= model_params['distance'][0].item() 
    dist= dist[0]
    
    cond= model_params['conditional'][0].item() 
    cond= cond[0]
    
    temp=int(model_params['temperature'][0][0])

    time_off=int(model_params['time_offsets'][0][0])

    max_iter=int(model_params['max_iter'][0][0])

    max_adapt_iter=int(model_params['max_adapt_iter'][0][0])

    b_size=int(model_params['batch_size'][0][0])

    learn_rate=int(model_params['learning_rate'][0][0])
    
    out_dim= int(model_params['output_dimension'][0][0])
    
    verb= model_params['verbose'][0].item() 
    verb= verb[0]
    
    n_h_u=int(model_params['num_hidden_units'][0][0])
    
    pad_before_transform_= model_params['pad_before_transform'][0].item() 
    # Converti il valore di 'pad_before_transform' in maiuscolo e rimuovi le virgolette
    if pad_before_transform_ == "True" or pad_before_transform_ == "true":
        p_b_t = True
    else:
        p_b_t = False
        
    

    
    hybrid_= model_params['hybrid'][0].item() 
    if hybrid_ == "True" or hybrid_== "true":
        hyb = True
    else:
        hyb = False


except FileNotFoundError:
   print("model_params file is missing!!!")


######################################## carico_dati #####################


# Verifica se PyTorch è stato compilato con il supporto GPU
if torch.cuda.is_available():
    # Stampa le informazioni sulla GPU disponibile
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
else:
    print("Nessuna GPU disponibile, si sta utilizzando la CPU.")



### importo dati hip e gorilla come benchamark

# ### dati neurali del ratto (spikes)
# hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
# hip_neur=hippocampus_pos.neural.numpy()
# hip_neur_tensor=hippocampus_pos.neural
# # ## Dati comportamentali (di cui non disponiamo)
# # #### posizione (valore unico); direzione (dummy 0 1)
# hip_behav=hippocampus_pos.continuous_index.numpy()
# hip_behav_tensor=hippocampus_pos.continuous_index


# # ###  CONFRONTI ###

# # if monkey_behav_index== monkey_behav_pos:
# #     print("Sono uguali")
# # else:
# #     print("Non sono uguali")


# # np.array_equal(monkey_neur_target ,monkey_neur)

# # if np.all(monkey_behav_index == monkey_behav_pos):
# #     print("Sono uguali")
# # else:
# #     print("Non sono uguali")

# # np.array_equal(behavior_data ,hip_behav)
# #np.allclose(neural_data ,hippocampus_pos.neural.numpy())


################################## MODELLISTICA ###############################

torch.cuda.empty_cache()
#### POSSO USARE TARGET COME VARIABILE AUSILIARIA
#max_iter=1000
#b_size= 2**8



#model_type='supervised'

cebra_target_model = CEBRA(model_architecture=mod_arch,
                           distance=dist,
                           conditional= cond,
                           temperature=temp,
                           time_offsets=time_off,
                           max_iterations=max_iter,
                           max_adapt_iterations=max_adapt_iter,
                           batch_size=b_size,
                           learning_rate=learn_rate,
                           output_dimension=out_dim,
                           verbose=verb,
                           num_hidden_units=n_h_u,
                           pad_before_transform=p_b_t,
                           hybrid=hyb,
                           device='cuda_if_available')


# cebra_target_model.fit(data_)
# cebra_target_model.fit(data_)

# intermediate_outputs = []

# def hook_fn(module, input, output):
#     intermediate_outputs.append(output)

# # Presupponendo che `model` sia il tuo modello PyTorch e `layer_name` sia il nome del modulo di cui vuoi catturare l'output.

# layer = cebra_target_model._modules.get(layer_name)
# hook = layer.register_forward_hook(hook_fn)

# # Ora esegui la forward pass
# output = model(input_data)

# # Non dimenticare di rimuovere l'hook una volta finito per evitare perdite di memoria.
# hook.remove()



def run_model(model, data, labels, model_type):
        if model_type == "hypothesis":
            # If the model is in supervised mode, use both data and labels
            model.fit(data, labels)
      
        elif model_type == "discovery":
            ### solo dati time
            model.fit(data)
        elif model_type == "shuffle":
            shuffled_labels = np.random.permutation(labels)
            model.fit(data, shuffled_labels)
         # Assicurati che il percorso alla directory 'data' esista

            
        joblib.dump(model, 'fitted_model.pkl')
    
    
        with torch.no_grad():
            return model.transform(data), model.model_.state_dict()
    
cebra_output, ceb_model = run_model(cebra_target_model, data_, label_, mod_type)

# numpy_dict = {key: value.numpy() for key, value in ceb_model.items()}
### generiamo e salviamo output nella directory ordianta d amatlab
cebra_mat={'cebra_output':cebra_output}
# savemat("model.mat", numpy_dict)

#numpy_output = cebra_output.detach().cpu().numpy() if torch.is_tensor(cebra_output) else cebra_output
numpy_model_ = {key.replace('.', '_'): value for key, value in ceb_model.items()}
numpy_model = {key: value.detach().cpu().numpy() for key, value in numpy_model_.items()}

#norms = np.linalg.norm(cebra_output, axis=1)
model_path = 'fitted_model.pkl'  # Se il modello è nella directory corrente, questo è sufficiente

# Scrivi il percorso in un file di testo
#with open('model_path.txt', 'w') as f:
 #   f.write(model_path)

scipy.io.savemat(os.path.join(output_directory, 'cebra_output.mat'), cebra_mat)
scipy.io.savemat(os.path.join(output_directory, 'model_struct.mat'), numpy_model)


### salvo dati neurali e behavior

#### codice per gestire memoria

# def can_process_entire_dataset(data, model_memory_footprint):
#     if not torch.cuda.is_available():
#         return False

#     data_memory = data.nbytes if isinstance(data, np.ndarray) else data.element_size() * data.nelement()
#     torch.cuda.empty_cache()
#     available_memory = torch.cuda.get_device_properties(0).total_memory
#     used_memory = torch.cuda.memory_allocated(0)
#     free_memory = available_memory - used_memory

#     return (data_memory + model_memory_footprint) < free_memory



#batch_size = 2**18 # potenze di 2 a seconda di quanto vogliamo stressare al gpu
# def run_model(model, data, labels=None):
#     if model_type == "supervised":
#         model.fit(data, labels)
#     else:
#         model.fit(data)

#     with torch.no_grad():
#         return model.transform(data)
      


# def process_data_in_batches(data, labels=None):
#     num_batches = len(data) // batch_size
#     batch_outputs = []

#     for i in range(num_batches):
#         batch_data = data[i * batch_size : (i + 1) * batch_size]
#         batch_labels = labels[i * batch_size : (i + 1) * batch_size] if model_type == "supervised" else None
#         batch_output = run_model(cebra_target_model, batch_data, batch_labels)
#         batch_outputs.append(batch_output)
#         del batch_data, batch_labels, batch_output
#         gc.collect()
#         torch.cuda.empty_cache()

#     if len(data) % batch_size != 0:
#         remaining_data = data[num_batches * batch_size :]
#         remaining_labels = labels[num_batches * batch_size :] if model_type == "supervised" else None
#         batch_output = run_model(cebra_target_model, remaining_data, remaining_labels)
#         batch_outputs.append(batch_output)

#     return np.concatenate(batch_outputs, axis=0)

# # Run the model in batches
# cebra_target = process_data_in_batches(data_, label_ if model_type == "supervised" else None)
