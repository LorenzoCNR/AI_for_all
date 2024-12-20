

#!pip install --pre 'cebra[datasets,demos]'
#!pip install ripser
import os
import sys
import warnings
import typing
#sys.path.append('/path/to/your/directory')
#sys.path.insert(0,'/path/to/your/directory')
###base_dir=r'/home/zlollo/CNR/Cebra_for_all'
#os.chdir(r'C:\Users\zlollo2\Desktop\Strila_27_03_24\CNR neuroscience\cebra_codes')
os.chdir(os.getcwd())
os.getcwd()
os.chdir(r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes')
#base_dir=r'/media/zlollo/STRILA/CNR neuroscience/codic_vari'
#base_dir=r'F:\CNR neuroscience'
import logging
import time
#os.chdir(base_dir)
import openTSNE
from joblib import Parallel, delayed
#import ripser
import numpy as np
import torch
import multiprocessing
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import random
import cebra.datasets
import umap
import pandas as pd
import matplotlib.pyplot as plt
import joblib as jl
import cebra.datasets
from scipy import stats
# import tensorflow as tf
from cebra import CEBRA
#from dataset import SingleRatDataset  
from matplotlib.collections import LineCollection
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import sklearn.metrics
#import inspect
#import torch
from cebra.datasets.hippocampus import *
import h5py
import pathlib
from matplotlib.markers import MarkerStyle
import logging
import seaborn as sns
#from tensorflow.python.client import device_lib

'''

Neural data is spike counts binned into 25ms time window and the behavior is 
position and the running direction (left, right) of a rat.
#the behavior label is structured as 3D array consists of position, right, and
left.The neural and behavior recordings are parsed into trials (a round trip 
from one end of the track) and the trials are split into a train, valid and
test set with k=3 nested cross validation.

'''

### put this plot in an extra file
def plot_hippocampus(ax, embedding, label, gray = False, idx_order = (0,1,2)):
    r_ind = label[:,1] == 1
    l_ind = label[:,2] == 1
    
    if not gray:
        r_cmap = 'cool'
        l_cmap = 'viridis'
        r_c = label[r_ind, 0]
        l_c = label[l_ind, 0]
    else:
        r_cmap = None
        l_cmap = None
        r_c = 'gray'
        l_c = 'gray'
    
    idx1, idx2, idx3 = idx_order
    r=ax.scatter(embedding [r_ind,idx1], 
               embedding [r_ind,idx2], 
               embedding [r_ind,idx3], 
               c=r_c,
               cmap=r_cmap, s=0.5)
    l=ax.scatter(embedding [l_ind,idx1], 
               embedding [l_ind,idx2], 
               embedding [l_ind,idx3], 
               c=l_c,
               cmap=l_cmap, s=0.5)
    
    #ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
   # ax.xaxis.pane.set_edgecolor('w')
   # ax.yaxis.pane.set_edgecolor('w')
  #  ax.zaxis.pane.set_edgecolor('w')
        
    return ax


#### function to create a specific db for rat split and seed
def _call_dataset(name, split_no, split, seed):
    DATA_NAME = f'rat-hippocampus-{name}-3fold-trial-split-{split_no}'
    
    dataset = cebra.datasets.init(DATA_NAME, split=split)
    #### inquire what offset is (some sort of temporal bias)
    dataset.offset.right = 5
    dataset.offset.left = 5
    return dataset

### what is the best k? 
def find_best_k(emb_train, emb_valid, label_train, label_valid):
    metric = 'cosine'
    # Possible values of k
    neighbors = np.power(np.arange(1, 6), 2) 
    best_score = float('-inf')
    best_k = None

    for n in neighbors:
        pos_decoder = KNeighborsRegressor(n, metric=metric)
        dir_decoder = KNeighborsClassifier(n, metric=metric)

        pos_decoder.fit(emb_train, label_train[:, 0])
        dir_decoder.fit(emb_train, label_train[:, 1])

        pos_pred = pos_decoder.predict(emb_valid)
        dir_pred = dir_decoder.predict(emb_valid)

        prediction = np.stack([pos_pred, dir_pred], axis=1)
        valid_score = sklearn.metrics.r2_score(label_valid[:, :2], prediction)

        if valid_score > best_score:
            best_score = valid_score
            best_k = n

    return best_k

def decoding_mod(emb_train, emb_test, label_train, label_test, n_neighbors):
    pos_decoder = KNeighborsRegressor(n_neighbors=n_neighbors, metric='cosine')
    dir_decoder = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    
    pos_decoder.fit(emb_train, label_train[:, 0])
    dir_decoder.fit(emb_train, label_train[:, 1])
    
    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)
    
    prediction = np.stack([pos_pred, dir_pred], axis=1)
    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(np.abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])
    
    return test_score, pos_test_err, pos_test_score, prediction


##############  MODELS

######################## CEBRA
## time params: {temp=3.213; offs_l =offs_r=5; time_off=10, max_iter=1000, #hid_un=64; l_r=0.003}
### behav params: {temp=1.266; offs_l =offs_r=5; time_off=10, max_iter=1000, #hid_un=64; l_r=0.002}

cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=512,
                        learning_rate=0.003,
                        temperature=3.213,
                        output_dimension=3,
                        max_iterations=2000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        num_hidden_units=64,
                        time_offsets=10,
                        hybrid = False)

###àà consider the call CEBRE(**model_params)


#### okkio che i parametri cambiano nelle vaie modalità
def decode_cebra(rat, split_no, seed):
    # hippocampus={}
    # hippocampus_dataset_name = 'rat-hippocampus-single-{}'.format(rat)
    # hippocampus[rat] = cebra.datasets.init(hippocampus_dataset_name)
    
    full_db=_call_dataset(rat,split_no, 'all',seed)
    train_set = _call_dataset(rat, split_no, 'train', seed)
    valid_set = _call_dataset(rat, split_no, 'valid', seed)
    test_set = _call_dataset(rat, split_no, 'test', seed)
      
    
    data_neural=full_db.neural.numpy()
    neural_train, label_train = train_set.neural.numpy(), train_set.index.numpy()
    neural_val, label_val = valid_set.neural.numpy(), valid_set.index.numpy()
    neural_test, label_test = test_set.neural.numpy(), test_set.index.numpy()
    
    #data_neural=hippocampus[rat].neural.numpy()
    ### CEBRA TIME fit on the entire db ###
    cebra_model.fit(data_neural)

    
    
    #cebra_model.fit(neural_train, label_train)
    cebra_train = cebra_model.transform(neural_train)
    cebra_val = cebra_model.transform(neural_val)
    cebra_test = cebra_model.transform(neural_test)
    
    ### (cfr article )

    best_k = find_best_k(cebra_train, cebra_val, label_train, label_val)

    # Evaluate on validation set
    val_score, val_pos_err, val_pos_score, _ = decoding_mod(cebra_train, cebra_val, label_train, label_val, best_k)

    # Evaluate on test set
    test_score, test_pos_err, test_pos_score, predictions = decoding_mod(cebra_train, 
                         cebra_test, label_train, label_test, best_k)
    

    return {
        'rat': rat,
        'split_no': split_no,
        'seed': seed,
        'k': best_k,
        'val_score': val_score,
        'val_pos_err': val_pos_err,
        'val_pos_score': val_pos_score,
        'test_score': test_score,
        'test_pos_err': test_pos_err,
        'test_pos_score': test_pos_score,
        'predictions': predictions.tolist()  # Convert to list for JSON serialization or similar
    }

# Tracking execution time
# start_time = time.time()

# results = []
# itera=1
# for rat in ["achilles", "buddy", "cicero", "gatsby"]:
#     for split_no in [0, 1, 2]:
#         for seed in range(10):
            
#             result = process_data(rat, split_no, seed)
#             results.append(result)
#             itera=itera+1
#             print(itera)

# cebra_decode_df = pd.DataFrame(results)
# print(decode_results_df)


if __name__ == "__main__":
    start_time = time.time()
    results = []
    for rat in ["achilles", "buddy", "cicero", "gatsby"]:
        for split_no in [0, 1, 2]:
            for seed in range(10):
                result = decode_cebra(rat, split_no, seed)
                if result:
                    results.append(result)
    cebra_decode_df = pd.DataFrame(results)
    print(cebra_decode_df)
        
    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time} seconds.")



#cebra_hyp_decode0=cebra_hyp_decode
#cebra_hyp_decode=cebra_decode_df


ceb1_stats=cebra_decode_df.describe()
ceb2_stats=cebra_hyp_decode.describe()

cebra_time_decode=cebra_decode_df

######## UMAP DECODING ######################

umap_model = umap.UMAP(
    n_neighbors=24,
    n_components=2,
    n_jobs=-1,
    min_dist=0.0001,
        #learning_rate='auto' if params['learning_rate'] == 'auto' else float(params['learning_rate'][0, 0]),
    random_state={'True': True, 'False': False, 'None': None}.get(False),
    metric='euclidean')

### consider the call umap.UMAP(**model_params)

def decode_umap(rat, split_no, seed):
    full_db=_call_dataset(rat,split_no, 'all',seed)
    data_neural=full_db.neural.numpy()
    
    train_set = _call_dataset(rat, split_no, 'train', seed)
    valid_set = _call_dataset(rat, split_no, 'valid', seed)
    test_set = _call_dataset(rat, split_no, 'test', seed)
        
    neural_train, label_train = train_set.neural.numpy(), train_set.index.numpy()
    neural_val, label_val = valid_set.neural.numpy(), valid_set.index.numpy()
    neural_test, label_test = test_set.neural.numpy(), test_set.index.numpy()
    
    ## check for sparsity issue (csr to lil)
    
    # if sp.issparse(data_neural):
    #   data_neural = data_neural.tolil()
    # if sp.issparse(neural_train):
    #   neural_train = neural_train.tolil()
    # if sp.issparse(neural_val):
    #   neural_val = neural_val.tolil()
    # if sp.issparse(neural_test):
    #    neural_test = neural_test.tolil()

    
    umap_model.fit(data_neural)
    
    umap_train = umap_model.transform(neural_train)
    umap_val = umap_model.transform(neural_val)
    umap_test = umap_model.transform(neural_test)
    

    best_k = find_best_k(umap_train, umap_val, label_train, label_val)

    # Evaluate on validation set
    val_score, val_pos_err, _, _ = decoding_mod(umap_train, umap_val, label_train, label_val, best_k)

    # Evaluate on test set
   # test_score, test_pos_err, test_pos_score, predictions = decoding_mod(cebra_train, 
    #                       cebra_test, label_train, label_test, best_k)
    test_score, test_pos_err, test_pos_score, predictions = decoding_mod(umap_train, 
                         umap_test, label_train, label_test, best_k)

    return {
        'rat': rat,
        'split_no': split_no,
        'seed': seed,
        'k': best_k,
        'val_score': val_score,
        'val_pos_err': val_pos_err,
        'test_score': test_score,
        'test_pos_err': test_pos_err,
        'test_pos_score': test_pos_score,
        'predictions': predictions.tolist()  
    }

# Tracking execution time normal method
if __name__ == "__main__":
    start_time = time.time()
    results = []
    for rat in ["achilles", "buddy", "cicero", "gatsby"]:
        for split_no in [0, 1, 2]:
            for seed in range(10):
                result = decode_umap(rat, split_no, seed)
                if result:
                    results.append(result)
    
    umap_decode_df = pd.DataFrame(results)
    print(umap_dec_df)
    
    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time} seconds.")


##### multiprocessing
def worker(args):
    rat, split_no, seed = args
    print(f"Inizio elaborazione per {rat}, Split: {split_no}, Seed: {seed}", flush=True)
    result = decode_umap(rat, split_no, seed)
    print(f"Completato: {rat}, Split: {split_no}, Seed: {seed}", flush=True)
    return result

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def worker(args):
#     rat, split_no, seed = args
#     logging.info(f"Start Processing {rat}, Split: {split_no}, Seed: {seed}")
#     result = decode_umap(rat, split_no, seed)
#     logging.info(f"Completed {rat}, Split: {split_no}, Seed: {seed}")
#     return result


def main():
    # Use 'spawn' o 'forkserver'
    multiprocessing.set_start_method('spawn')  
    start_time = time.time()
    tasks = [(rat, split_no, seed) for rat in ["achilles", "buddy", "cicero", 
                                               "gatsby"]
                                      for split_no in [0, 1, 2]
                                      for seed in range(10)]
    
    num_cores = multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_cores // 2) as pool:
        results = pool.map(worker, tasks)
        pool.close()
        pool.join()

    if results:
        umap_decode_df = pd.DataFrame(results)
        print(tsne_decode_df)
    else:
        print("No results to display, some errors occurred.")

    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time} seconds.")

if __name__ == "__main__":
    main()


#umap_decode_0=decode_results_df

######################################## TSNE °°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°°

tsne_model = openTSNE.TSNE(
    n_components=2,
    perplexity=10,
    learning_rate='auto',
    n_iter=1000,
    initialization='pca',
    metric='cosine',
    early_exaggeration=16.44,
    theta=0.5,
    verbose=True)
### consider the call umap.UMAP(**model_params)

def decode_tsne(rat, split_no, seed):
    full_db=_call_dataset(rat,split_no, 'all',seed)
    data_neural=full_db.neural.numpy()
    
    train_set = _call_dataset(rat, split_no, 'train', seed)
    valid_set = _call_dataset(rat, split_no, 'valid', seed)
    test_set = _call_dataset(rat, split_no, 'test', seed)
        
    neural_train, label_train = train_set.neural.numpy(), train_set.index.numpy()
    neural_val, label_val = valid_set.neural.numpy(), valid_set.index.numpy()
    neural_test, label_test = test_set.neural.numpy(), test_set.index.numpy()
    

    
    model_fit=tsne_model.fit(data_neural)
    
    tsne_train = model_fit.transform(neural_train)
    tsne_val = model_fit.transform(neural_val)
    tsne_test = model_fit.transform(neural_test)
    

    best_k = find_best_k(tsne_train, tsne_val, label_train, label_val)

    # Evaluate on validation set
    val_score, val_pos_err, _, _ = decoding_mod(tsne_train, tsne_val, label_train, label_val, best_k)

    # Evaluate on test set
   # test_score, test_pos_err, test_pos_score, predictions = decoding_mod(cebra_train, 
    #                       cebra_test, label_train, label_test, best_k)
    test_score, test_pos_err, test_pos_score, predictions = decoding_mod(tsne_train, 
                         tsne_test, label_train, label_test, best_k)

    return {
        'rat': rat,
        'split_no': split_no,
        'seed': seed,
        'k': best_k,
        'val_score': val_score,
        'val_pos_err': val_pos_err,
        'test_score': test_score,
        'test_pos_err': test_pos_err,
        'test_pos_score': test_pos_score,
        'predictions': predictions.tolist()  
    }

# Tracking execution time
if __name__ == "__main__":
    start_time = time.time()
    results = []
    itera=0
    for rat in ["achilles", "buddy", "cicero", "gatsby"]:
        for split_no in [0, 1, 2]:
            for seed in range(10):
                result = decode_tsne(rat, split_no, seed)
                if result:
                    results.append(result)
                    itera=itera+1
                    print(itera)
    
    tsne_decode_df = pd.DataFrame(results)
    print(tsne_dec_df)
    
    elapsed_time = time.time() - start_time
    print(f"Execution Time: {elapsed_time} seconds.")

tsne_decode_df0_df=tsne_decode_df

################################################# PLOTS and tables

#####
decoding_results={'cebra_time': cebra_time_decode,'cebra_behav': cebra_hyp_decode,
                  'tsne': tsne_decode_df, 'umap': umap_decode_0         }

#### ["achilles", "buddy", "cicero", "gatsby"]:

def get_metrics(results):
    for key, results_ in results.items():
        df = results_.copy()
        df["method"] = key
        df["test_pos_err"] *= 100
        # Assicurati che il filtraggio corrisponda ai tuoi dati, ad esempio:
        df = df[df.rat == "cicero"].pivot_table(
            "test_pos_err", index=("method", "seed"), aggfunc="mean"
        )
        yield df

df_decoding_results = pd.concat(get_metrics(decoding_results)).reset_index()

##àà BoxPlot
plt.figure(figsize=(2, 2), dpi=200)
ax = plt.gca()
show_boxplot(
    df=df_decoding_results,
    metric="test_pos_err",
    ax=ax,
    color="C1",
    labels=[
        
        "cebra_behav",
        "cebra_time",
        "tsne",
        "umap"
    ],
)
ticks = [0, 10, 20, 30, 40, 50, 65]
ax.set_xlim(min(ticks), max(ticks))
ax.set_xticks(ticks)
ax.set_xlabel("Error [cm]")
ax.set_yticklabels(
    [
        "CEBRA-Behavior",
        "CEBRA-Time",
        "t-SNE",
        "UMAP",
     
    ]
)
# plt.savefig("figure2d.svg", bbox_inches = "tight", transparent = True)
plt.show()

### 
cebra_hyp_decode.to_csv('cebra_hyp_decode.csv', index=False)
cebra_time_decode.to_csv('cebra_time_decode.csv', index=False)
umap_decode_0.to_csv('umap_decode.csv', index=False)
tsne_decode_df.to_csv('tsne_decode.csv', index=False)


##### MEAN VARIANCE COMPARISON AND ANOVA
def get_metrics2(results):
    all_metrics = []  
    anova_data = []  
    #
    for key, results_ in results.items():
        df = results_.copy()
        df["method"] = key
        df["test_pos_err"] *= 100

        df_filtered = df[df.rat == "cicero"]
        stats_df = df_filtered.groupby("method")['test_pos_err'].agg(['mean', 'var', 'count'])

        all_metrics.append(stats_df)
        
        if not df_filtered.empty:
           anova_data.append(df_filtered['test_pos_err'])

        combined_df = pd.concat(all_metrics)
    
    anova_result = None
    if len(anova_data) >= 2:
        anova_result = stats.f_oneway(*anova_data)

    return combined_df, anova_result

anova_df_achille,_=get_metrics2(decoding_results)
anova_df_gatsby,_=get_metrics2(decoding_results)
anova_df_buddy,_=get_metrics2(decoding_results)
anova_df_cicero,_=get_metrics2(decoding_results)

anova_df_achille.to_csv("tab_achille.csv", index=False)
anova_df_gatsby.to_csv("tab_gatsby.csv", index=False)

anova_df_buddy.to_csv("tab_buddy.csv", index=False)

anova_df_cicero.to_csv("tab_cicero.csv", index=False)



#test_set.index.numpy()

# tr0_np=train_set.index.numpy()
# te0_np=test_set.index.numpy()
# va0_np=valid_set.index.numpy()


# tr1_np=train_set.index.numpy()
# te1_np=test_set.index.numpy()
# va1_np=valid_set.index.numpy()

# tr2_np=train_set.index.numpy()
# te2_np=test_set.index.numpy()
# va2_np=valid_set.index.numpy()


##############  DECODING 

##### TRAIN TEST SPLIT (And Validation)
# def split_data(data, test_ratio):

#     split_idx = int(len(data)* (1-test_ratio))
#     neural_train = data.neural[:split_idx]
#     neural_test = data.neural[split_idx:]
#     label_train = data.continuous_index[:split_idx]
#     label_test = data.continuous_index[split_idx:]

#     return neural_train.numpy(), neural_test.numpy(), label_train.numpy(), label_test.numpy()



# def split_data(data, test_ratio, validation_ratio):
#     # Compute objext sizes given the shares
#     total_size = len(data)
#     test_size = int(total_size * test_ratio)
#     validation_size = int(total_size * validation_ratio)
#     train_size = total_size - test_size - validation_size

#     # Indici di fine per ciascun set
#     train_end = train_size
#     val_end = train_size + validation_size

#     # Assegna i dati ai rispettivi set mantenendo l'ordine temporale
#     neural_train = data.neural[:train_end]
#     neural_val = data.neural[train_end:val_end]
#     neural_test = data.neural[val_end:]

#     label_train = data.continuous_index[:train_end]
#     label_val = data.continuous_index[train_end:val_end]
#     label_test = data.continuous_index[val_end:]

#     return neural_train.numpy(), neural_val.numpy(), neural_test.numpy(),label_train.numpy(), label_val.numpy(), label_test.numpy()



######################N.B!!!! CORRECT SPLITTING IN TIME SERIES ########################
### ROLLING WINDOWS
# def rolling_window_split(data, train_size, val_size, test_size):
#     import torch
#     total_size = len(data)
#     start_point = 0

#     while start_point + train_size + val_size + test_size <= total_size:
#         train_slice = data[start_point:start_point + train_size]
#         val_slice = data[start_point + train_size:start_point + train_size + val_size]
#         test_slice = data[start_point + train_size + val_size:start_point + train_size + val_size + test_size]
        
#         if isinstance(data, torch.Tensor):
#             # Convert tensor to NumPy if it's a tensor, handle CUDA case
#             train_slice = train_slice.cpu().numpy() if train_slice.is_cuda else train_slice.numpy()
#             val_slice = val_slice.cpu().numpy() if val_slice.is_cuda else val_slice.numpy()
#             test_slice = test_slice.cpu().numpy() if test_slice.is_cuda else test_slice.numpy()
        
#         yield train_slice, val_slice, test_slice
#         start_point += test_size

####INCREMENTAL TRAIN TEST SPLIT
# def incremental_train_test_split(data, initial_train_size, step_size, val_size, test_size):
#     import torch
#     total_size = len(data)
#     end_train = initial_train_size
    
#     while end_train + val_size + test_size <= total_size:
#         train_slice = data[:end_train]
#         val_slice = data[end_train:end_train + val_size]
#         test_slice = data[end_train + val_size:end_train + val_size + test_size]
        
#         if isinstance(data, torch.Tensor):
#             # Convert tensor to NumPy if it's a tensor, handle CUDA case
#             train_slice = train_slice.cpu().numpy() if train_slice.is_cuda else train_slice.numpy()
#             val_slice = val_slice.cpu().numpy() if val_slice.is_cuda else val_slice.numpy()
#             test_slice = test_slice.cpu().numpy() if test_slice.is_cuda else test_slice.numpy()
        
#         yield train_slice, val_slice, test_slice
#         end_train += step_size

# for train_data, val_data, test_data in rolling_window_split(data, 100, 50, 30):
#     print("Train data length:", len(train_data), "Validation data length:", len(val_data), "Test data length:", len(test_data))
# ######################################################################################################################

