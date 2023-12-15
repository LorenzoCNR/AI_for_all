# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')
import os
os.getcwd()
import sys
#### Mettere la directory di interesse la stessa di matlab
from pathlib import Path
#base_dir=r'/home/zlollo/CNR/Cebra_for_all'
#os.chdir(base_dir)
import time

#!pip install ripser
#import ripser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from cebra import CEBRA
from scipy.io import loadmat
from scipy.io import savemat
#from dataset import SingleRatDataset  # Assumendo che il codice sia in 'dataset.py'
from matplotlib.collections import LineCollection
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
import inspect
import torch
from cebra.datasets.hippocampus import *
import tensorflow as tf



def run_hip_models(base_path):

    os.chdir(base_path)
    ######################### DA CAMBIARE ##################################
    hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')

           #from tensorflow.python.client import device_lib
    
    '''
    # Verifica la disponibilità di GPU
    if tf.test.is_gpu_available():
        print("TensorFlow sta utilizzando una GPU.")
    else:
        print("TensorFlow non sta utilizzando una GPU.")
    
    
    print(device_lib.list_local_devices())
    
    def get_available_devices():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos]
    with tf.device('/device:GPU:0'):
        print(get_available_devices())
    '''
    
    IMAGES_PATH = Path() / "images" 
    IMAGES_PATH.mkdir(parents=True, exist_ok=True)
    
    def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
        path = IMAGES_PATH / f"{fig_id}.{fig_extension}"
        if tight_layout:
            plt.tight_layout()
        plt.savefig(path, format=fig_extension, dpi=resolution)
    
    
    
    ###### Load model hyperparameters
    try:
    
        data_param=loadmat('params.mat')
        
        params = data_param['params']
        
        mod_arch= params['mod_arch'][0].item() 
        mod_arch= mod_arch[0]
        
        out_dim= int(params['output_dimension'][0][0])
        
        temp=int(params['temperature'][0][0])
        
        max_iter=int(params['max_iter'][0][0])
        
        dist= params['distance'][0].item() 
        dist= dist[0]
        
        cond= params['conditional'][0].item() 
        cond= cond[0]
        
        time_off=int(params['time_offsets'][0][0])
    
    except:
        mod_arch='offset10-model'
        out_dim=3
        temp=1
        max_iter=10000
        dist='cosine'
        cond='time_delta'
        time_off=10
    
    
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
    
    
    fig = plt.figure(figsize=(12,4), dpi=150)
    plt.subplots_adjust(wspace = 0.3)
    ax = plt.subplot(121)
    ax.imshow(hippocampus_pos.neural.numpy()[:1000].T, aspect = 'auto', cmap = 'gray_r')
    plt.ylabel('Neuron #')
    plt.xlabel('Time [s]')
    plt.xticks(np.linspace(0,1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))
    
    ax2 = plt.subplot(122)
    ax2.scatter(np.arange(1000), hippocampus_pos.continuous_index[:1000,0], c = 'gray', s=1)
    plt.ylabel('Position [m]')
    plt.xlabel('Time [s]')
    plt.xticks(np.linspace(0, 1000, 5), np.linspace(0, 0.025*1000, 5, dtype = int))
    save_fig("Neural Data and Bheavior", tight_layout=False,fig_extension="pdf")
    plt.show()
    
    
    
    #max_iterations = 10 ## defaut 5000
    
    neural_data=hippocampus_pos.neural
    behavior_data=hippocampus_pos.continuous_index.numpy()
    
    
    behavior_dic={'dir':behavior_data[:,0],
                  'right':behavior_data[:,1],
                  'left':behavior_data[:,2]}
    
    savemat('beavior_data.mat',behavior_dic )
    
    output_dimension = 32 #here, we set as a variable for hypothesis testing below.
    
    #max_iter=4000
    cebra_posdir3_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp, 
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_posdir3_model.fit(neural_data,behavior_data)
    cebra_posdir3 = cebra_posdir3_model.transform(neural_data)
    
    #d_vis_hyp=data["visualization"]['hypothesis']
    
    
    hypoth={"embedding": cebra_posdir3, "label": behavior_data}
    
    ### Qui si allena in modello con shuffled neural datao
    
    cebra_posdir_shuffled3_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    hippocampus_shuffled_posdir = np.random.permutation(behavior_data)
    cebra_posdir_shuffled3_model.fit(neural_data, hippocampus_shuffled_posdir)
    cebra_posdir_shuffled3 = cebra_posdir_shuffled3_model.transform(neural_data)
    
    shuff={"embedding": cebra_posdir_shuffled3, "label": behavior_data}
    
    
    #### modello con time ma senza info comportamentalii
    ### setto conditional su time
    ### OKKIO CHE QUI TEMPERATURE è tipo 1.12 (12% maggiore)
    ### ...okkio che condtionale è time, non time delta
    cebra_time3_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional='time',
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_time3_model.fit(neural_data)
    cebra_time3 = cebra_time3_model.transform(neural_data)
    
    ttime={"embedding": cebra_time3, "label": behavior_data}
    
    
    ### modello ibrido con info temporali e posizionali 
    cebra_hybrid_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=out_dim,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off,
                            hybrid = True)
    
    cebra_hybrid_model.fit(neural_data, behavior_data)
    cebra_hybrid = cebra_hybrid_model.transform(neural_data)
    
    hhybrid={"embedding": cebra_hybrid, "label": behavior_data}
    
    
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
    # test_ratio=0.2
    # split_idx = int(len(neural_data)* (1-test_ratio)) 
    # neural_train = neural_data[:split_idx]
    # neural_test = neural_data[split_idx:]
    # label_train = behavior_data[:split_idx]
    # label_test = behavior_data[split_idx:]
       
    def split_data(data, test_ratio):
    
        split_idx = int(len(data)* (1-test_ratio))
        neural_train = data.neural[:split_idx]
        neural_test = data.neural[split_idx:]
        label_train = data.continuous_index[:split_idx]
        label_test = data.continuous_index[split_idx:]
        
        return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()
    
    neural_train, neural_test, label_train, label_test = split_data(hippocampus_pos, 0.2) 
    
    cebra_posdir_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_pos_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_dir_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
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
    
    cebra_posdir_shuffled_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_pos_shuffled_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
                            conditional=cond,
                            device='cuda_if_available',
                            verbose=True,
                            time_offsets=time_off)
    
    cebra_dir_shuffled_model = CEBRA(model_architecture=mod_arch,
                            batch_size=512,
                            learning_rate=3e-4,
                            temperature=temp,
                            output_dimension=output_dimension,
                            max_iterations=max_iter,
                            distance=dist,
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
    ### okkio che sul transform qui usa tutti i dati
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
        'loss_pos_dir': loss_pos_dir[0].numpy() if isinstance(loss_pos_dir[0], 
                                    torch.Tensor) else loss_pos_dir[0],
        'loss_pos': loss_pos.numpy() if isinstance(loss_pos, torch.Tensor) else loss_pos,
        'loss_dir': loss_dir.numpy() if isinstance(loss_dir, torch.Tensor) else loss_dir,
        'loss_pos_dir_shuffle': loss_pos_dir_shuffle.numpy() if isinstance(loss_pos_dir_shuffle, 
                                torch.Tensor) else loss_pos_dir_shuffle,
        'loss_pos_shuffle': loss_pos_shuffle.numpy() if isinstance(loss_pos_shuffle,
                                torch.Tensor) else loss_pos_shuffle,
        'loss_dir_shuffle': loss_dir_shuffle.numpy() if isinstance(loss_dir_shuffle,
                                    torch.Tensor) else loss_dir_shuffle
    }
    
    savemat('model_loss.mat', models_loss)
    
    
    
    #### ddecoding...in poche parole rendiamo utili le info degli embedding generati
    ### ottenendo tipo info su posizione e direzione di un soggetto.
    ### diciamo che è una  forma di validazione del modello
    ### si usa il knn 
    
    
    def decoding_pos_dir(emb_train, emb_test, label_train, label_test, n_neighbors=36):
        pos_decoder = KNeighborsRegressor(n_neighbors, metric = 'cosine')
        dir_decoder = KNeighborsClassifier(n_neighbors, metric = 'cosine')
        
        pos_decoder.fit(emb_train, label_train[:,0])
        dir_decoder.fit(emb_train, label_train[:,1])
        
        pos_pred = pos_decoder.predict(emb_test)
        dir_pred = dir_decoder.predict(emb_test)
        
        prediction =np.stack([pos_pred, dir_pred],axis = 1)
    
        test_score = sklearn.metrics.r2_score(label_test[:,:2], prediction)
        pos_test_err = np.median(abs(prediction[:,0] - label_test[:, 0]))
        pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:,0])
        
        return test_score, pos_test_err, pos_test_score, prediction
    
    #### occhio che ..._decode...sono errori
    
    _, cebra_posdir_decode,_,pred_posdir_decode = decoding_pos_dir(cebra_posdir_train,
                                cebra_posdir_test, label_train, label_test)
    _, cebra_pos_decode,_,pred_pos_decode = decoding_pos_dir(cebra_pos_train, 
                           cebra_pos_test, label_train, label_test)
    _,cebra_dir_decode,_,pred_dir_decode = decoding_pos_dir(cebra_dir_train, 
                         cebra_dir_test, label_train, label_test)
    _,cebra_posdir_shuffled_decode,_,pred_posdir_shuffled_decode = decoding_pos_dir(cebra_posdir_shuffled_train,
                 cebra_posdir_shuffled_test, label_train, label_test)
    _,cebra_pos_shuffled_decode,_,pred_pos_shuffle_decode = decoding_pos_dir(cebra_pos_shuffled_train,
                 cebra_pos_shuffled_test, label_train, label_test)
    _,cebra_dir_shuffled_decode,_,pred_dir_shuffled_decode = decoding_pos_dir(cebra_dir_shuffled_train, 
                 cebra_dir_shuffled_test, label_train, label_test)
    
    
    cebra_4th_output_decoding={'cebra_posdir_decode': cebra_posdir_decode,
                        'cebra_pos_decode':cebra_pos_decode,
                        'cebra_dir_decode':cebra_dir_decode,
                        'cebra_posdir_shuffled_decode': cebra_posdir_shuffled_decode,
                        'cebra_pos_shuffled_decode':cebra_pos_shuffled_decode,
                        'cebra_dir_shuffled_decode':cebra_dir_shuffled_decode,
                        'cebra_posdir_loss': cebra_posdir_model.state_dict_['loss'].numpy(),
                       'cebra_pos_loss': cebra_pos_model.state_dict_['loss'].numpy(),
                       'cebra_dir_loss': cebra_dir_model.state_dict_['loss'].numpy(),
                       'cebra_posdir_shuffled_loss':cebra_posdir_shuffled_model.state_dict_['loss'].numpy(),
                       'cebra_pos_shuffled_loss':cebra_pos_shuffled_model.state_dict_['loss'].numpy(),
                       'cebra_dir_shuffled_loss':cebra_dir_shuffled_model.state_dict_['loss'].numpy(),
                       'cebra_posdir_test':cebra_posdir_test,
                        'label_test':label_test,
                        'pred_posdir_decode': pred_posdir_decode,
                        'pred_posdir_shuffled_decode':pred_posdir_shuffled_decode
                      
                        }
    
    savemat('cebra_decoding.mat', cebra_4th_output_decoding)
    
    
    #visualiz_0=pd.DataFrame[(hypoth, shuff, ttime,hhybrid)]
    
    
    dd_names = ['Hypothesis: position', 'Shuffled Labels', 'Discovery: time only', 
                'Hybrid: time + behavior', 'viz',
    
                   'Behavior_Topology', 'loss', 'Circular_coord']
    
    dd= {'visualization': [pd.NA] * len(dd_names), 'topology':
         [pd.NA] * len(dd_names),'hypothesis_testing': [None] * len(dd_names)}
    
    dd = pd.DataFrame(dd, index=dd_names)
    
    dd['visualization']['Hypothesis: position']=hypoth
    dd['visualization']['Shuffled Labels']=shuff
    dd['visualization']['Discovery: time only']=ttime
    dd['visualization']['Hybrid: time + behavior']=hhybrid
    
    dd.at["viz","hypothesis_testing"]={}
    dd.at["viz","hypothesis_testing"]['dir']=cebra_dir_all
    dd.at["viz","hypothesis_testing"]['dir-shuffled']=cebra_dir_shuffled_all
    dd.at["viz","hypothesis_testing"]['pos']=cebra_pos_all
    dd.at["viz","hypothesis_testing"]['pos-shuffled']=cebra_pos_shuffled_all
    dd.at["viz","hypothesis_testing"]['posdir']=cebra_posdir_all
    dd.at["viz","hypothesis_testing"]['posdir-shuffled']=cebra_posdir_shuffled_all
     
     
    dd.at["loss","hypothesis_testing"]={}
    dd.at["loss","hypothesis_testing"]['dir']=cebra_dir_model.state_dict_['loss'].numpy()
    dd.at["loss","hypothesis_testing"]['dir-shuffled']=cebra_dir_shuffled_model.state_dict_['loss'].numpy()
    dd.at["loss","hypothesis_testing"]['pos']=cebra_pos_model.state_dict_['loss'].numpy()
    dd.at["loss","hypothesis_testing"]['pos-shuffled']=cebra_pos_shuffled_model.state_dict_['loss'].numpy()
    dd.at["loss","hypothesis_testing"]['posdir']=cebra_posdir_model.state_dict_['loss'].numpy()
    dd.at["loss","hypothesis_testing"]['posdir-shuffled']=cebra_posdir_shuffled_model.state_dict_['loss'].numpy()
     
    
    err_loss={ 'error_posdir_decode':cebra_posdir_decode,
               'error_pos_decode':cebra_pos_decode,
               'error_dir_decode':cebra_dir_decode,
               'error_posdir_decode_shuffled':cebra_posdir_shuffled_decode,
               'error_pos_decode_shuffled':cebra_pos_shuffled_decode,
               'error_dir_decode_shuffled':cebra_dir_shuffled_decode,
               'loss_posdir_decode':cebra_posdir_model.state_dict_['loss'][-1],
               'loss_pos_decode':cebra_pos_model.state_dict_['loss'][-1],
               'loss_dir_decode':cebra_dir_model.state_dict_['loss'][-1],
               'loss_posdir_decode_shuffled':cebra_posdir_shuffled_model.state_dict_['loss'][-1],
               'loss_pos_decode_shuffled':cebra_pos_shuffled_model.state_dict_['loss'][-1],
               'loss_dir_decode_shuffled':cebra_dir_shuffled_model.state_dict_['loss'][-1]}
    
    err_={'median_error':[], 'loss_end':[]}
    err_['median_error'].append(label_test)
    
   
    
    mod1_pred={'cebra_posdir_test':cebra_posdir_test,
              'label_test':label_test,
              'pred_posdir_decode':pred_posdir_decode,
              'pred_posdir_shuffled_decode':    pred_posdir_shuffled_decode}


    
    
    return  dd, err_loss, mod1_pred
if __name__ == "__main__":
    run_hip_models()




#### plot in matlab

