# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#!pip install --pre 'cebra[dev,demos]'

#!pip uninstall -y dreimac
#!pip uninstall -y cebra
#!pip uninstall -y ripser

import os
os.getcwd()
################################## CAMBIARE ########################################
main_folder=r'/home/zlollo/CNR/git_out_cebra/elab_Mirco'

os.chdir(main_folder)
import sys

#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')


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

from cebra.datasets.hippocampus import *
import sklearn.metrics
import scipy.io
from scipy.io import savemat
from scipy.io import loadmat
import gc
import torch
#import tensorflow as tf

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics



##################### PArametri Generici ###########################àssss

# try:
    
    
    
#     general_param=loadmat('general_params.mat')
        
#     gen_params= general_param['general_params']
        
#     main_fold= gen_params['main_fold'][0].item() 
#     main_fold= main_fold[0]
    
#     mod_type= gen_params['model_type'][0].item() 
#     mod_type= mod_type[0]
    
#     # cond= model_params['conditional'][0].item() 
#     # cond= cond[0]
    
#     # temp=int(model_params['temperature'][0][0])

# except FileNotFoundError:
#    print("general_params file is missing!!!")


# paths_to_remove = ['../third_party','~CNR/Cebra_for_all/third_party',
#  'Cebra_for_all/third_party', '/home/zlollo/CNR/third_party','-./third_party',
#  '/home/zlollo/CNR/Cebra_for_all/third_party/pivae']

# # Remove all paths in the list from sys.path
# for path in paths_to_remove:
#     while path in sys.path:
#         sys.path.remove(path)

# path_Mirco=r'/home/zlollo/CNR/git_out_cebra/elab_Mirco'

# sys.path.insert(0, path_Mirco)

# #sys.path.insert(0, path_Mirco)
# #sys.path.insert(0, '/home/zlollo/CNR/Cebra_for_all/third_party/pivae')
# sys.path.append(path_Mirco)
# sys.path





###############à###### Load model hyperparameters #############################
# base_path=loadmat('path_to_save.mat', struct_as_record=False)

# base_dir=base_path['main_folder'][0]

# os.chdir(base_dir)


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
 
    
# except:
#     mod_arch='offset10-model'
#     out_dim=3
#     temp=1
#     max_iter=200
#     dist='cosine'
#     cond='time_delta'
#     time_off=10

except FileNotFoundError:
   print("model_params file is missing!!!")

######################################## carico_dati #####################



### dati Mirco
#dati_M_path='/home/zlollo/CNR/git_out_cebra/elab_Mirco/data_norm_long.mat'
data_mat = scipy.io.loadmat('data_norm_long.mat')
data_norm_long=data_mat['data_norm_long'].astype('float32')

#print(data_mat.keys())

data_=data_norm_long[:,0:-1].astype('float32')
label_=data_norm_long[:,-1].astype('float32')


# Verifica se PyTorch è stato compilato con il supporto GPU
if torch.cuda.is_available():
    # Stampa le informazioni sulla GPU disponibile
    print(f"GPU disponibile: {torch.cuda.get_device_name(0)}")
else:
    print("Nessuna GPU disponibile, si sta utilizzando la CPU.")



### importo dati hip e gorilla come benchamark
# ############## DATI RATTO #################################
# ### dati neurali del ratto (spikes)
# hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
# hip_neur=hippocampus_pos.neural.numpy()
# hip_neur_full=hippocampus_pos.neural
# ## Dati comportamentali (di cui non disponiamo)
# #### posizione (valore unico); direzione (dummy 0 1)
# hip_behav=hippocampus_pos.continuous_index.numpy()

# ############## DATI SCIMMIA #################################
# monkey_pos = cebra.datasets.init('area2-bump-pos-active')
# monkey_target = cebra.datasets.init('area2-bump-target-active')

# # ## dati neurali scimmia
# monkey_neur=monkey_pos.neural.numpy()
# monkey_neur_full=monkey_pos.neural

# # ### dati posizione comportamento(due posizioni nel tensore ma sembrano uguali)
# monkey_behav_index=monkey_pos.continuous_index.numpy()
# monkey_behav_pos=monkey_pos.pos.numpy()


# # ###n dati target scimmia
# # ### i dati targett includono anche un discrete index 
# monkey_neur_target=monkey_target.neural.numpy()
# monkey_neur_full_target=monkey_target.neural

# ### check sui dati behavior
# ### la variabile target è un vettore con valori 0-7...secondo la direzione
# monkey_behav_target=monkey_target.discrete_index.numpy()

# monkey_behav_target.min()


# ###  CONFRONTI ###

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
#### qui i parametri li devo dare da MatLab
l_max=int(label_.max())
l_min=int(label_.min())

torch.cuda.empty_cache()
#### POSSO USARE TARGET COME VARIABILE AUSILIARIA
#max_iter=1000
#b_size= 2**8


#model_type='supervised'

cebra_target_model = CEBRA(model_architecture=mod_arch,
                           distance=dist,
                           conditional='time_delta',
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
                           hybrid=True,
                           device='cuda_if_available')






    
def run_model(model, data, labels, model_type):
    if model_type == "supervised":
        # If the model is in supervised mode, use both data and labels
        model.fit(data, labels)
    else:
        # If the model is in unsupervised mode, use only data
        model.fit(data)

    with torch.no_grad():
        return model.transform(data)
 
cebra_output = run_model(cebra_target_model, data_, label_, mod_type)

cebra_mat={'cebra_output':cebra_output}
savemat('cebra_output.mat', cebra_mat)

  
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