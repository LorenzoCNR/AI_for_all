import openTSNE
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

def run_model(model_type, params, data_, subtrain=False):
    """
    Runs the given model type (tsne, umap, cebra, etc.) on the provided data.
    """

    X_train = X_valid = y_train = y_valid = None
    X_sub_train = y_sub_train = None

    # Assegna i valori per train, validation e subtrain
    for key, value in data_.items():
        if re.match(r'X_train', key):
            X_train = value
        elif re.match(r'X_val', key):
            X_valid = value
        elif re.match(r'y_train', key):
            y_train = value
        elif re.match(r'y_val', key):
            y_valid = value
        elif re.match(r'X_subtrain', key):
            X_sub_train = value
        elif re.match(r'y_subtrain', key):
            y_sub_train = value

    # Controlla se i dati sono presenti
    if X_train is None or X_valid is None or y_train is None or y_valid is None:
        print("Required training/validation data not found.")
        return None, None, None, None, None  # Return fewer values to avoid unpacking error

    model = create_model(model_type, params)

    # Modello con subtrain (in base ai dati già divisi)
    if model_type in ['cebra_time', 'umap', 'tsne']:
        fitted_model = model.fit(X_train)
        embeddings_train = fitted_model.transform(X_train)
        embeddings_sub_train = fitted_model.transform(X_sub_train)
        embeddings_valid = fitted_model.transform(X_valid)
    elif model_type in ['cebra_behavior', 'cebra_hybrid']:
        fitted_model = model.fit(X_train, y_train)
        embeddings_train = fitted_model.transform(X_train)
        embeddings_sub_train = fitted_model.transform(X_sub_train)
        embeddings_valid = fitted_model.transform(X_valid)
    
    return embeddings_train, embeddings_valid, embeddings_sub_train, fitted_model, y_train, y_valid, y_sub_train, X_sub_train

