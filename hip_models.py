# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import os
os.getcwd()
import sys
#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')

base_dir=r'/home/zlollo/CNR/Cebra_for_all'


os.chdir(base_dir)
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
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import scipy.io
from scipy.io import savemat
#import inspect
import torch
#from cebra.datasets.hippocampus import *

### i dati direzione posizione sono 
data_mat = scipy.io.loadmat('data.mat')


neural_data = data_mat['neuralDataMat']
behavior_data = data_mat['behaviorDataMat']


'''
Cebra- behaviour  addestrea un modello con output 3d che usa info posiionali (posiz e direz)
con l'uso di una variabili ausiliaria, time_delta, il tempo durante il training
del modello
Il modello Cebra-shuffled, viene usato come coontrollo 
Il modello Cebra-time utilizza solo info  temporali  e non comportamentali 
(posizionali)
Cebra hybrid infine utilizza una combinazione di informazioni temporali e
 comportamentali in modo più integrato e bilanciato.
Invece di trattare le informazioni temporali come un semplice contesto 
ausiliario, "CEBRA-Hybrid" fonde queste informazioni con le variabili 
comportamentali (posizione e direzione) per costruire un modello più complesso 
che tiene conto sia del comportamento che del tempo in maniera paritaria.

-output dimension è la dimensione dello spazio embedded di proiezione
- distance misura di similarità scelta
- temeprature: parametro è comunemente usato nelle funzioni di loss basate 
    su softmax, come la InfoNCE loss. La temperatura modifica la distribuzione
    delle probabilità, controllando il livello di enfasi sui dati positivi 
    rispetto a quelli negativi durante l'addestramento. 
    Un valore di temperatura più alto rende la distribuzione più uniforme (
    meno enfasi sugli esempi positivi), mentre un valore più basso rende 
    la distribuzione più concentrata sugli esempi positivi.

- conditional:Indica una condizione o una caratteristica specifica usata 
  dal modello. Nel caso di 'time_delta', verosimilmente il modello 
  considera le differenze temporali tra i campioni come parte del suo
  processo di apprendimento.
  
- time_offsets: Questo parametro potrebbe essere utilizzato per specificare
 un intervallo temporale entro il quale i dati vengono considerati nel modello.
 Per esempio, un offset di 10 potrebbe indicare che il modello considera
 i dati entro 10 unità temporali dall'evento di riferimento.

'''



max_iterations = 10 ## defaut 5000
output_dimension = 32 #here, we set as a variable for hypothesis testing below.


cebra_posdir3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1, 
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_posdir3_model.fit(neural_data,behavior_data[:,0])
cebra_posdir3 = cebra_posdir3_model.transform(neural_data)


### Qui si allena in modello con shuffled neural datao

cebra_posdir_shuffled3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

hippocampus_shuffled_posdir = np.random.permutation(behavior_data)
cebra_posdir_shuffled3_model.fit(neural_data, hippocampus_shuffled_posdir)
cebra_posdir_shuffled3 = cebra_posdir_shuffled3_model.transform(neural_data)

#### modello con time ma senza info comportamentalii
### setto conditional su time
###
cebra_time3_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1.12,
                        output_dimension=3,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_time3_model.fit(behavior_data)
cebra_time3 = cebra_time3_model.transform(behavior_data)

### modello ibrido con info temporali e posizionali 
cebra_hybrid_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=4,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10,
                        hybrid = True)

cebra_hybrid_model.fit(neural_data, behavior_data)
cebra_hybrid = cebra_hybrid_model.transform(neural_data)

############################## Questo in matlab ###############################

# output model with 3d output that use positional info
cebra_posdir3
## output mdello con shuffled neural datao
cebra_posdir_shuffled3 
#### output modello con time ma senza info comportamentalii
cebra_time3 
### output modello ibrido con info temporali e posizionali 
cebra_hybrid

cebra_1step_output={'cebra_posdir3': cebra_posdir3,
                    'cebra_posdir_shuffled3':cebra_posdir_shuffled3,
                    'cebra_time3':cebra_time3,
                    'cebra_hybrid': cebra_hybrid}

#cebra_1step_output_clean = {key.replace(' ', '_'): value for key, value in cebra_1step_output.items()}
### salva in formato mat
savemat('cebra_1step_output.mat',cebra_1step_output)

#### visualiziamo gli spazi embedded generati con CEBRA su MATLAB (cfr funz
### plot hippocampus)

###############################################################################


################################### In PYTHON #################################
'''### Test delle Ipotesi: Addestramento di modelli con diverse ipotesi 
 sull'encoding posizionale dell'ippocampo. L'obiettivo è confrontare
 diversi modelli CEBRA-Behavior addestrati utilizzando diverse variabili 
 comportamentali: solo la posizione, solo la direzione, entrambe queste 
 variabili, e modelli di controllo con variabili comportamentali mescolate
 casualmente (shuffled).
Si utilizza  quindi una dimensione del modello predefinita. 
Nel lavoro originale descritto nel documento, sono state utilizzate dimensioni 
del modello che variano da 3 a 64 per analizzare i dati dell'ippocampo, e si è 
osservata una topologia coerente attraverso queste diverse dimensioni.
Per le analisi di decodifica successive, si utilizzerà un set di dati diviso, 
con l'80% dei dati train e il 20% test. 
Solo il train set sarà  per addestrare i modelli.
In sintesi, lo scopo di questo esperimento è di esplorare come differenti tipi
 di variabili comportamentali (posizione, direzione, entrambe, o mescolate) 
 influenzino la capacità dei modelli CEBRA-Behavior di codificare informazioni
 sull'ippocampo. 
 '''
#def split_data(data, test_ratio):
test_ratio=0.2
split_idx = int(len(neural_data)* (1-test_ratio)) 
neural_train = neural_data[:split_idx]
neural_test = neural_data[split_idx:]
label_train = behavior_data[:split_idx]
label_test = behavior_data[split_idx:]
    

cebra_posdir_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_dir_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        device='cuda_if_available',
                        verbose=True)

### alleniamo modelli con posizione direzione o entrambe
### otteniamo train e test set embeddinbg

cebra_posdir_model.fit(neural_train, label_train)
###
cebra_posdir_train = cebra_posdir_model.transform(neural_train)
cebra_posdir_test = cebra_posdir_model.transform(neural_test)

cebra_pos_model.fit(neural_train, label_train[:,0])
###
cebra_pos_train = cebra_pos_model.transform(neural_train)
cebra_pos_test = cebra_pos_model.transform(neural_test)


cebra_dir_model.fit(neural_train, label_train[:,1])
###
cebra_dir_train = cebra_dir_model.transform(neural_train)
cebra_dir_test = cebra_dir_model.transform(neural_test)


##àà Alleniamo modelli di controllo con shuffled behaviour variables

cebra_posdir_shuffled_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_pos_shuffled_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)

cebra_dir_shuffled_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=3e-4,
                        temperature=1,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        device='cuda_if_available',
                        verbose=True)


### generiamo etichette shuffled per il train set

shuffled_posdir = np.random.permutation(label_train)
shuffled_pos = np.random.permutation(label_train[:,0])
shuffled_dir = np.random.permutation(label_train[:,1])


# Train the models with shuffled behavior variables and get train/test embeddings
cebra_posdir_shuffled_model.fit(neural_train, shuffled_posdir)
##à# 
cebra_posdir_shuffled_train = cebra_posdir_shuffled_model.transform(neural_train)
cebra_posdir_shuffled_test = cebra_posdir_shuffled_model.transform(neural_test)

cebra_pos_shuffled_model.fit(neural_train, shuffled_pos)
##à# 
cebra_pos_shuffled_train = cebra_pos_shuffled_model.transform(neural_train)
cebra_pos_shuffled_test = cebra_pos_shuffled_model.transform(neural_test)

cebra_dir_shuffled_model.fit(neural_train, shuffled_dir)
##à# 
cebra_dir_shuffled_train = cebra_dir_shuffled_model.transform(neural_train)
cebra_dir_shuffled_test = cebra_dir_shuffled_model.transform(neural_test)

##############################à MATLAB ###########################################
### vediamo gli embeddings da differenti ipotesi
##à# MATLAB
cebra_pos_all = cebra_pos_model.transform(neural_data)
cebra_dir_all = cebra_dir_model.transform(neural_data)
cebra_posdir_all = cebra_posdir_model.transform(neural_data)
##à# MATLAB
cebra_pos_shuffled_all = cebra_pos_shuffled_model.transform(neural_data)
cebra_dir_shuffled_all = cebra_dir_shuffled_model.transform(neural_data)
cebra_posdir_shuffled_all = cebra_posdir_shuffled_model.transform(neural_data)

cebra_2nd_output_hyp_test={'cebra_pos_all': cebra_pos_all,
                    'cebra_dir_all':cebra_dir_all,
                    'cebra_posdir_all':cebra_posdir_all,
                    'cebra_pos_shuffled_all': cebra_pos_shuffled_all,
                    'cebra_dir_shuffled_all':cebra_dir_shuffled_all,
                    'cebra_posdir_shuffled_all':cebra_posdir_shuffled_all
                    }

#cebra_1step_output_clean = {key.replace(' ', '_'): value for key, value in cebra_1step_output.items()}
### salva in formato mat
savemat('cebra_2nd_output_hyp_test.mat',cebra_2nd_output_hyp_test)

############## PLOT results in  MATLAB ##################à


### vediamo la perdita di modelli allenati secondo le diverse ipotesi in 
### MATLAB
############ python

loss_pos_dir=cebra_posdir_model.state_dict_['loss'],
loss_pos=cebra_pos_model.state_dict_['loss']
loss_dir=cebra_dir_model.state_dict_['loss']
loss_pos_dir_shuffle=cebra_posdir_shuffled_model.state_dict_['loss']
loss_pos_shuffle=cebra_pos_shuffled_model.state_dict_['loss']
loss_dir_shuffle=cebra_dir_shuffled_model.state_dict_['loss']


models_loss = {
    'loss_pos_dir': loss_pos_dir[0].numpy() if isinstance(loss_pos_dir[0], torch.Tensor) else loss_pos_dir[0],
    'loss_pos': loss_pos.numpy() if isinstance(loss_pos, torch.Tensor) else loss_pos,
    'loss_dir': loss_dir.numpy() if isinstance(loss_dir, torch.Tensor) else loss_dir,
    'loss_pos_dir_shuffle': loss_pos_dir_shuffle.numpy() if isinstance(loss_pos_dir_shuffle, torch.Tensor) else loss_pos_dir_shuffle,
    'loss_pos_shuffle': loss_pos_shuffle.numpy() if isinstance(loss_pos_shuffle, torch.Tensor) else loss_pos_shuffle,
    'loss_dir_shuffle': loss_dir_shuffle.numpy() if isinstance(loss_dir_shuffle, torch.Tensor) else loss_dir_shuffle
}

savemat('model_loss.mat', models_loss)

fig = plt.figure(figsize=(5,5))
ax = plt.subplot(111)
ax.plot(cebra_posdir_model.state_dict_['loss'], c='deepskyblue', label = 'position+direction')
ax.plot(cebra_pos_model.state_dict_['loss'], c='deepskyblue', alpha = 0.3, label = 'position')
ax.plot(cebra_dir_model.state_dict_['loss'], c='deepskyblue', alpha=0.6,label = 'direction')
ax.plot(cebra_posdir_shuffled_model.state_dict_['loss'], c='gray', label = 'pos+dir, shuffled')
ax.plot(cebra_pos_shuffled_model.state_dict_['loss'], c='gray', alpha = 0.3, label = 'position, shuffled')
ax.plot(cebra_dir_shuffled_model.state_dict_['loss'],c='gray', alpha=0.6,label = 'direction, shuffled')








ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlabel('Iterations')
ax.set_ylabel('InfoNCE Loss')
plt.legend(bbox_to_anchor=(0.5,0.3), frameon = False )
plt.show()

#### ddecoding...fare update.....
