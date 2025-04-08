

#!pip install --pre 'cebra[datasets,demos]'
#!pip install ripser
import os
import sys
import warnings
import typing
os.chdir(os.getcwd())
os.getcwd()

#### windows
base_dir=r'F:\CNR_neuroscience\cebra_git'
## ubuntu
#base_dir=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git'

os.chdir(base_dir)

import logging
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import openTSNE
import scipy.sparse as sp
import sympy
from joblib import Parallel, delayed
#import ripser
import numpy as np
import torch
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
import seaborn as sns
from scipy.sparse import lil_matrix
#from tensorflow.python.client import device_lib

'''

Neural data is spike counts binned into 25ms time window and the behavior is 
position and the running direction (left, right) of a rat.
#the behavior label is structured as 3D array consists of position, right, and
left.The neural and behavior recordings are parsed into trials (a round trip 
from one end of the track) and the trials are split into a train, valid and
test set with k=3 nested cross validation.

'''
### UBUNTU
# sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party')
# sys.path.insert(0, '/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/third_party/pivae')

### WINDOWS
# Correct the paths by removing the leading backslash if you're specifying an absolute path
sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party')
sys.path.insert(0, 'F:\\CNR_neuroscience\\cebra_git\\Cebra_for_all\\third_party\\pivae')


import pivae_code.datasets 
import pivae_code.conv_pi_vae
import pivae_code.pi_vae



# offset_right = 5
# offset_left = 5
# def _call_dataset(name, split_no, split, seed):
#      DATA_NAME = f'rat-hippocampus-{name}-3fold-trial-split-{split_no}'
    
#      dataset = cebra.datasets.init(DATA_NAME, split=split)
#      #### inquire what offset is (some sort of temporal bias)
#      dataset.offset.right = offset_right
#      dataset.offset.left = offset_left
#      return dataset

off_r = 5
off_l = 5

def load_dataset_for_rat(rat, split_no, split_type='all', seed=None):
    """
    Args:
        rat (str)
        split_no (int)
        split_type (str): Tipo di split, default è 'all'.
        seed (int): optional seed; default none. 

    """
    dataset_name = f'rat-hippocampus-{rat}-3fold-trial-split-{split_no}'
    dataset = cebra.datasets.init(dataset_name, split=split_type)

    # Configure offset 
    if hasattr(dataset, 'offset'):
        dataset.offset.right = off_r
        dataset.offset.left = off_l

    return dataset



# class BatchGenerator:
#     def __init__(self, data, labels, batch_size=128):
#         assert len(data) == len(labels), "Data and labels must have the same length."
#         self.data = data
#         self.labels = labels
#         self.batch_size = batch_size
#         self.num_batches = math.ceil(len(data) / batch_size)
#         self.index = 0

#     def next(self):
#         if self.index >= len(self.data):  # Reset index if all data has been used
#             self.index = 0
#             raise StopIteration("All batches have been generated.")
        
#         end_index = min(self.index + self.batch_size, len(self.data))
#         batch_data = self.data[self.index:end_index]
#         batch_labels = self.labels[self.index:end_index]
#         self.index = end_index
#         return batch_data, batch_labels

#     def __iter__(self):
#         return self

#     def __next__(self):
#         return self.next()

### what is the best k? 

def find_best_k(emb_train, emb_valid, label_train, label_valid):
    metric = 'cosine'
    # Possible values of k
    neighbors = np.power(np.arange(1, 6), 2)
    best_score = float('-inf')
    best_k = None
    best_pos_err = None
    best_pos_score = None
    
    for n in neighbors:
        pos_decoder = KNeighborsRegressor(n_neighbors=n, metric=metric)
        dir_decoder = KNeighborsClassifier(n_neighbors=n, metric=metric)
        
        pos_decoder.fit(emb_train, label_train[:, 0])
        dir_decoder.fit(emb_train, label_train[:, 1])
        
        pos_pred = pos_decoder.predict(emb_valid)
        dir_pred = dir_decoder.predict(emb_valid)
        
        prediction = np.stack([pos_pred, dir_pred], axis=1)
        valid_score = sklearn.metrics.r2_score(label_valid[:, :2], prediction)
        pos_err = np.median(np.abs(pos_pred - label_valid[:, 0]))
        pos_score = sklearn.metrics.r2_score(label_valid[:, 0], pos_pred)
        
        if valid_score > best_score:
            best_score = valid_score
            best_k = n
            best_pos_err = pos_err
            best_pos_score = pos_score
    
    return best_k, best_score, best_pos_err, best_pos_score

def decoding_mod(pos_decoder, dir_decoder, emb_test, label_test):
    """
    Use pre-trained decoders to predict and evaluate on the test set.
    """
    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)
    
    prediction = np.stack([pos_pred, dir_pred], axis=1)
    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(np.abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])
    
    return test_score, pos_test_err, pos_test_score, prediction

###### split data in batches to make em more derivable (per conv_pivae)
def custom_data_generator(x_all, u_one_hot):
    while True:
        for ii in range(len(x_all)):
            yield ([x_all[ii], u_one_hot[ii]], None)


def make_loader(X,y, batch_size):
    def _to_batch_list(x, y, batch_size):
        if x is not None and y is not None:
            x = x.squeeze()
            if len(x.shape) == 3:
                x = x.transpose(0,2,1)
            x_batch_list = np.array_split(x, int(len(x) / batch_size))
            y_batch_list = np.array_split(y, int(len(y) / batch_size))
        else:
            return None, None
       
        return x_batch_list, y_batch_list
    
    x_batches, y_batches = _to_batch_list(X, y, batch_size)

    # if x_batches is None or y_batches is None:
    #    return None, None, None

    loader = custom_data_generator(x_batches, y_batches)
    return x_batches, y_batches, loader



##############################  MODELS ##########################################

######################## CEBRA
## time params: {temp=3.213; offs_l =offs_r=5; time_off=10, max_iter=1000, #hid_un=64; l_r=0.003}
### behav params: {temp=1.266; offs_l =offs_r=5; time_off=10, max_iter=1000, #hid_un=64; l_r=0.002}

cebra_model = CEBRA(model_architecture='offset10-model',
                        batch_size=256,
                        learning_rate=0.003,
                        temperature=1,
                        output_dimension=3,
                        max_iterations=1000,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        num_hidden_units=32,
                        time_offsets=10,
                        hybrid = False)

### consider the call CEBRE(**model_params)


########################### UMAP 
umap_model = umap.UMAP(
    n_neighbors=24,
    n_components=2,
    n_jobs=-1,
    min_dist=0.0001,
    n_epochs=1000,
        #learning_rate='auto' if params['learning_rate'] == 'auto' else float(params['learning_rate'][0, 0]),
    #random_state=42,
    metric='euclidean')

########################### TSNE

tsne_model = openTSNE.TSNE(
    n_components=2,
    perplexity=10,
    learning_rate='auto',
    n_iter=1000,
    initialization='pca',
    metric='cosine',
    early_exaggeration=16.44,
    theta=0.5,
    n_jobs=-1,
    negative_gradient_method='fft' ,
    verbose=True)

### consider the call umap.UMAP(**model_params)


#################### CONV-PIVAE ##############################


####################################### function for model choice
def initialize_model(model_type):
    if model_type == 'cebra_time' or model_type == 'cebra_behavior':
        return cebra_model 
    elif model_type == 'umap':
        return umap_model 
    elif model_type == 'tsne':
        return tsne_model
    elif model_type == 'conv_pivae':
        return pivae_code.conv_pi_vae.conv_vae_mdl(
                        dim_x=X.shape[1],
                        dim_z=2,
                        dim_u=3,
                        time_window=10,
                        gen_nodes=60,
                        n_blk=2,
                        mdl="poisson",
                        disc=False,
                        learning_rate=0.00025)
    


#### okkio che i parametri cambiano nelle vaie modalità

rat = 'achilles'
split_no = 0
data_ = load_dataset_for_rat(rat, split_no)

X=data_.neural.numpy()
y=data_.continuous_index.numpy()


###### ottimizzazione del dato
# X.dtype
# X.nbytes
# X_bit=X.astype(np.uint8)
# X_bit.nbytes

# y.nbytes
# y.dtype
#print(np.array_equal(X,X_bit))

def main(model_type, X,y, num_iterations=30):
    num_iterations = 30
    results = []
    
    
    if model_type not in ['cebra_time', 'cebra_behavior', 'umap', 'tsne',
                          'conv_pivae']:
      raise ValueError("Unsupported model type specified.")

    model = initialize_model(model_type)
    ### work on matrix sparsity

    #X_sp=sp.csr_matrix(X)
    #X_sp=sp.lil_matrix(X)

    for i in range(num_iterations):
        print(f"\n Starting {i+1}/{num_iterations}")
        # First split in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=i)
        
        # Further split of train data into internal train and validation
        X_train_internal, X_valid, y_train_internal, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=i)

        # Train model (e.g., CEBRA, UMAP) on internal training set
        print("Train Model...")
        #For CEBRA time or unsupervised models like UMAP/TSNE
        if model_type in ['cebra_time', 'umap', 'tsne']:
            inner_fit=model.fit(X_train)  
            emb_train_internal = inner_fit.transform(X_train_internal)
            emb_valid = inner_fit.transform(X_valid)
            outer_fit=inner_fit
            emb_train = outer_fit.transform(X_train)
            emb_test = outer_fit.transform(X_test)
        # CEBRA behaviour wants labels            
        elif model_type == 'cebra_behavior':
            inner_fit=model.fit(X_train, y_train) 
            emb_train_internal = inner_fit.transform(X_train_internal)
            emb_valid = inner_fit.transform(X_valid)
            outer_fit=inner_fit
            emb_train = outer_fit.transform(X_train)
            emb_test = outer_fit.transform(X_test)
        elif model_type== 'conv_pivae':
            batch_size = 200
            #i=43
            data_ = load_dataset_for_rat('achilles', 0)
            X_,y = data_[torch.arange(len(data_))].numpy(), data_.index.numpy()
            X_train, X_test, y_train, y_test = train_test_split(X_, y, test_size=0.30, random_state=i)
            
            # Further split of train data into internal train and validation
            X_train_internal, X_valid, y_train_internal, y_valid = train_test_split(X_train, y_train, test_size=0.15, random_state=i)

            train_X, train_y, train_loader = make_loader(X_train,y_train, batch_size)
            #valid_x, valid_u, valid_loader = make_loader(valid_set, batch_size)
            test_X, test_y, test_loader = make_loader(X_test, y_test, batch_size)
            train_X_internal, train_y_internal, train_loader_internal = make_loader(X_train_internal,y_train_internal, batch_size)
            #valid_x, valid_u, valid_loader = make_loader(valid_set, batch_size)
            valid_X, valid_y, valid_loader = make_loader(X_valid, y_valid, batch_size)

            inner_fit= model.fit(x=train_loader,
                     steps_per_epoch=len(train_X), epochs=1000,
                     verbose=1)
                     #validation_data=valid_loader,
                     #validation_steps=validation_steps,
                     #callbacks=[mcp])
            
            
            train_outputs_internal =model.predict([np.concatenate(train_X_internal),
                            np.concatenate(train_y_internal)])
            
            valid_outputs = model.predict([np.concatenate(valid_X),
                            np.concatenate(valid_y)])
            
            
            train_outputs =model.predict([np.concatenate(train_X),
                            np.concatenate(train_y)])
            
            emb_train_internal = train_outputs_internal[6]
            emb_valid = valid_outputs[6] 
            emb_train = train_outputs[6]
                    
            
            
            test_outputs =model.predict([np.concatenate(test_X),
                            np.concatenate(test_y)])
            
            emb_test = test_outputs[6]
            
    
        # Find best k using validation set
        print("Find best k using internal train and validation sets...")
        best_k, val_score, val_pos_err, val_pos_score = find_best_k(emb_train_internal, emb_valid, y_train_internal, y_valid)
        print(f"Best k is: {best_k} with validation score: {val_score}")

        # Re-train model on the entire training dataset
        # print("Re-train model on entire training dataset...")
        # if model_type in ['cebra_time', 'umap', 'tsne']:
        #     outer_fit=model.fit(X_train)  # Re-fit using the entire training dataset
        # elif model_type == 'cebra_behavior':
        # outer_fit=model.fit(X_train, y_train)
      

        # Configure decoders
        print("Decoders Config...")
        pos_decoder = KNeighborsRegressor(n_neighbors=best_k, metric='cosine')
        dir_decoder = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
        pos_decoder.fit(emb_train, y_train[:, 0])
        dir_decoder.fit(emb_train, y_train[:, 1])

        # Performance evaluation on the test set
        print("Performance evaluation on test set...")
        
        test_score, test_pos_err, test_pos_score, predictions = decoding_mod(pos_decoder, dir_decoder, emb_test, y_test)

        # Organize output for each iteration
        result = {
            'rat': 'rat_id',
            'split_no': i,
            'seed': 42 + i,
            'k': best_k,
            'val_score': val_score,
            'val_pos_err': val_pos_err,
            'val_pos_score': val_pos_score,
            'test_score': test_score,
            'test_pos_err': test_pos_err,
            'test_pos_score': test_pos_score,
            'predictions': predictions.tolist()
        }
        results.append(result)

    return results


if __name__ == "__main__":
    model_type = 'conv_pivae'  # Can be changed to cebra_behavior, umap, or tsne
    results_conv_pivae= main(model_type, X,y,30)
    print("Completed processing for:", model_type)

#results_cebra_behav=results_cebra_hybrid
#print(torch.cuda.is_available())  # Mostra True se CUDA è disponibile
#print(torch.version.cuda)      

################################################# PLOTS and tables

#####
cebra_time=pd.DataFrame(results_cebra_time)
cebra_behav=pd.DataFrame(results_cebra_behav)
cebra_hybrid=pd.DataFrame(results_cebra_hybrid)
cebra_hybrid2=pd.DataFrame(results_cebra_hybrid_std)

tsne_=pd.DataFrame(results_tsne)
umap_=pd.DataFrame(results_umap)


c_pivae=pd.DataFrame(results_conv_pivae)


decoding_results={'cebra_time': cebra_time,'cebra_behav': cebra_behav,
                 'cebra_hybrid': cebra_hybrid,
                 'tsne': tsne_ , 'umap': umap_ ,
                 #'cebra_hybrid2': cebra_hybrid2,
                 'conv_pivae': c_pivae}


def get_metrics(results):
    for key, results_ in results.items():
        df = results_.copy()
        df["method"] = key
        df["test_pos_err"] *= 100
        # Assicurati che il filtraggio corrisponda ai tuoi dati, ad esempio:
        df.pivot_table(
            "test_pos_err", index=("method", "seed"), aggfunc="mean"
        )
        yield df

df_decoding_results = pd.concat(get_metrics(decoding_results)).reset_index()

##àà BoxPlot

def show_boxplot(df, metric, ax, labels=None, color="C1"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sns.boxplot(
            data=df,
            y="method",
            x=metric,
            orient="h",
            order=labels,
            width=0.5,
            color="k",
            linewidth=2,
            flierprops=dict(alpha=0.5, markersize=0, marker=".", linewidth=0),
            medianprops=dict(
                c=color, markersize=0, marker=".", linewidth=2, solid_capstyle="round"
            ),
            whiskerprops=dict(solid_capstyle="butt", linewidth=0),
            showbox=False,
            showcaps=False,
            ax=ax,
        )
        marker_style = MarkerStyle("o", "none")
        sns.stripplot(
            data=df,
            y="method",
            x=metric,
            orient="h",
            size=4,
            color="k",
            order=labels,
            marker=marker_style,
            linewidth=1,
            ax=ax,
            alpha=0.75,
            jitter=0.15,
            zorder=-1,
        )
        ax.set_ylabel("")
        sns.despine(left=True, bottom=False, ax=ax)
        ax.tick_params(
            axis="x", which="both", bottom=True, top=False, length=5, labelbottom=True
        )
        return ax




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
        "cebra_hybrid",
        "tsne",
        "umap",
        'conv_pivae'
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
        "CEBRA-Hybrid",
        "t-SNE",
        "UMAP",
        "Conv PIVAE"
     
    ]
)
# plt.savefig("figure2d.svg", bbox_inches = "tight", transparent = True)
plt.show()


cebra_time.to_csv('cebra_time_decode.csv', index=False)
cebra_hybrid.to_csv('cebra_hybrid_decode.csv', index=False)
cebra_behav.to_csv('cebra_behav_decode.csv', index=False)
umap_.to_csv('umap_decode.csv', index=False)
tsne_.to_csv('tsne_decode.csv', index=False)
#cebra_hybrid2.to_csv('cebra_hybrid2_decode.csv', index=False)

conv_pivae


##### ANOVA e post hoc (Pairwise comparisons)

def get_metrics2(results):
    all_metrics = []  
    anova_data = []  
    tukey_labels= []
    #
    for key, results_ in results.items():
        df = results_.copy()
        df["method"] = key
        df["test_pos_err"] *= 100
        
        stats_df = df.groupby("method")['test_pos_err'].agg(['mean', 'var', 'count'])

        all_metrics.append(stats_df)
        
        if not df.empty:
           anova_data.append(df['test_pos_err'])
           tukey_labels.extend([key] * len(df))


        combined_df = pd.concat(all_metrics)
    anova_result = None
    if len(anova_data) >= 2:
        combined_anova_data = pd.concat(anova_data)
        anova_result = stats.f_oneway(*anova_data)
        combined_anova_df = pd.DataFrame({
            "Statistic": [anova_result.statistic],
            "p-value": [anova_result.pvalue]
        })
        
        if anova_result.pvalue < 0.05:
                tukey_result = pairwise_tukeyhsd(combined_anova_data, tukey_labels,
                                                 alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey_result.summary().data[1:],
                                        columns=tukey_result.summary().data[0])
                tukey_df.to_csv("tukey_results.csv")
            

    return combined_df, combined_anova_df, tukey_df if 'tukey_result' in locals() else None
    

stats_achille,anova_achille, tukey_achille =get_metrics2(decoding_results)

anova_achille.to_csv('anova_achille.csv',index=True)
stats_achille.to_csv('mean_var_models.csv', index=True)
tukey_achille.to_csv('anova_pairwise.csv', index=False)

