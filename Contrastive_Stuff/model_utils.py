#import openTSNE
import umap
import cebra
from joblib import Parallel, delayed
import numpy as np
import torch
import re
import multiprocessing
from numpy.lib.stride_tricks import as_strided
from multiprocessing import Pool
import random
import cebra.datasets
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import umap
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from cebra import CEBRA

def create_model(model_type, params, **extra_params):
    if model_type == 'tsne':
        return openTSNE.TSNE(**params)
    elif model_type == 'umap':
        return umap.UMAP(**params)
    elif model_type in ['cebra_time', 'cebra_behavior', 'cebra_hybrid']:
        return CEBRA(**params)
    elif model_type == 'conv_pivae':
        all_params = {**params, **extra_params}
        return pivae_code.conv_pi_vae.conv_vae_mdl(**all_params)
    else:
        raise ValueError("Unsupported model type")



# def run_model(model_type, params, data_, subtrain=False):
#     """
#     Runs the given model type (tsne, umap, cebra, etc.) on the provided data.
#     """
#     X_train = X_valid = y_train = y_valid = None
#     X_sub_train = y_sub_train = None

#     # Assign values for train, validation, and subtrain (if provided)
#     for key, value in data_.items():
#         if '_train' in key and 'X' in key:
#             X_train = value
#         elif '_train' in key and 'y' in key:
#             y_train = value
#         elif '_val' in key and 'X' in key:
#             X_valid = value
#         elif '_val' in key and 'y' in key:
#             y_valid = value
#         elif '_subtrain' in key and 'X' in key:
#             X_sub_train = value
#         elif '_subtrain' in key and 'y' in key:
#             y_sub_train = value

#     # Check if the main data (train/validation) is present
#     if X_train is None or X_valid is None or y_train is None or y_valid is None:
#         print("Required training/validation data not found.")
#         return None, None, None, None, None, None, None, None

#     model = create_model(model_type, params)

#     # Fit model and transform data
#     if model_type in ['cebra_time', 'umap', 'tsne']:
#         fitted_model = model.fit(X_train)
#         train_loss= fitted_model.state_dict_['loss']
#         embeddings_train = fitted_model.transform(X_train)
#         embeddings_valid = fitted_model.transform(X_valid)
#         fitted_all=model.fit[data_['X']]
#         embeddings_all=fitted_all.transform['y']
#         # Only transform `X_sub_train` if it's available
#         embeddings_sub_train = fitted_model.transform(X_sub_train) if X_sub_train is not None else None

#     elif model_type in ['cebra_behavior', 'cebra_hybrid']:
#         fitted_model = model.fit(X_train, y_train)
#         train_loss= fitted_model.state_dict_['loss']
#         embeddings_train = fitted_model.transform(X_train)
#         embeddings_valid = fitted_model.transform(X_valid)
        
#         # Only transform `X_sub_train` if it's available
#         embeddings_sub_train = fitted_model.transform(X_sub_train) if X_sub_train is not None else None

#     # Return results, handling `None` for `embeddings_sub_train` if `subtrain` data was not provided
#     return embeddings_train, embeddings_valid, embeddings_sub_train, fitted_model, fitted_all, y_train, y_valid, y_sub_train, X_sub_train, train_loss


def run_model(model_type, params, data_, train_data=None, transform_data=None, save_results=False, save_path="results/", return_keys=None):
    """
    Flexible function to train and transform data using a given model type
    (e.g., tsne, umap, cebra).

    Parameters:
        model_type (str): The type of model to use
        (e.g., 'tsne', 'umap', 'cebra').
        params: yaml external source.
        data_ (dict): Dictionary containing datasets (e.g., X_train, y_train,
                                                      etc.).
       - train_data (list, optional): List of keys in `data_` to use for training.
       - transform_data (list, optional): List of keys in `data_` to transform
        after training.
       - save_results (bool, optional): If True, saves the results to the specified path.
       - save_path (str, optional): Path to save results if `save_results` is True.
        return_keys (list, optional): Keys to include in the returned dictionary.

    Returns:
        dict: Dictionary containing the requested outputs.
    """

    #if save_results and not os.path.exists(save_path):
    #    os.makedirs(save_path)

    # Initialize variables
    results = {}
    X_train, y_train = None, None

    # Assign train data
    if train_data:
        X_train = data_.get(train_data[0])
        y_train = data_.get(train_data[1]) if len(train_data) > 1 else None

    # Check train data existence
    if X_train is None:
        raise ValueError("Training data is missing or invalid.")

    # Create and fit model
    model = create_model(model_type, params)
    if model_type in ['cebra_behavior', 'cebra_hybrid'] and y_train is not None:
        fitted_model = model.fit(X_train, y_train)
    else:
        fitted_model = model.fit(X_train)
    
    results['fitted_model']=fitted_model
    # Save training loss if available
    if hasattr(fitted_model, 'state_dict_') and 'loss' in fitted_model.state_dict_:
        results['train_loss'] = fitted_model.state_dict_['loss']

    # Assign transform data and generate embeddings
    if transform_data:
        for key in transform_data:
            X = data_.get(key)
            if X is not None:
                results[f'embeddings_{key}'] = fitted_model.transform(X)

    # Save results
    # if save_results:
    #     import pickle
    #     with open(os.path.join(save_path, f'{model_type}_model.pkl'), 'wb') as f:
    #         pickle.dump(fitted_model, f)
    #     for key, value in results.items():
    #         if 'embeddings' in key:
    #             np.save(os.path.join(save_path, f'{key}.npy'), value)

    # Filter output by requested keys
    if return_keys:
        return {key: results[key] for key in return_keys if key in results}

    return results
