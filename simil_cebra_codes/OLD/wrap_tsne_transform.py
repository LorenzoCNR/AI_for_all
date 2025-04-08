# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 15:27:25 2024

@author: zlollo2
"""

#os.chdir(r'C:\Users\zlollo2\Desktop\Strila_27_03_24\CNR neuroscience\cebra_codes')
from scipy.io import loadmat, savemat
import joblib as jl
import openTSNE
import os
import scipy
import sys


def load_model(model_file, data):
    # Carica il modello addestrato
    return jl.load(model_file), loadmat(data)

def transform_data(model, data_mat):
    
    transformed_data = model.transform(data_mat['data_n'])
    return transformed_data


#model, data=load_model('tsne_model.pkl', 'rat_n' )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python transform.py <model_file> <input_directory> <output_directory>")
        sys.exit(1)

    model_file, input_directory, output_directory = sys.argv[1:4]

    print(output_directory)
    
    # Carica il modello
    model, data_ = load_model(model_file,os.path.join(input_directory,'data_n.mat'))

    # Trasforma i dati
    transformed_data = transform_data(model,data_)
    
    transform_mat = {'transformed_data': transformed_data}
    
    try:
        # Tentativo di trasformazione e salvataggio dei dati
        scipy.io.savemat(os.path.join(output_directory, 'transf_data.mat'), transform_mat)
    except Exception as e:
        print(f"Error during saving file: {e}")
    
