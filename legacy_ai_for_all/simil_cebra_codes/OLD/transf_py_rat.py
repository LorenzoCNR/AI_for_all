import sys
import scipy.io

from scipy.io import loadmat, savemat

import joblib
import sklearn
import os
#print(f"Directory corrente: {os.getcwd()}")

output_directory='F:\CNR neuroscience\cebra_codes'

def load_model(model_path, data):
    # Carica il modello addestrato
    return joblib.load(model_path), loadmat(data)

def transform_data(model, data_mat):
    
    # Implementa la logica per caricare e trasformare i dati
    # Esempio: trasforma i dati utilizzando il modello
    transformed_data = model.transform(data_mat['data_n']) # Assumi che il modello abbia un metodo transform
    return transformed_data

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python transform_model.py <model_path> <data_path> <output_directory>")
        sys.exit(1)

    model_path, data_path, output_directory = sys.argv[1:4]

    print(output_directory)
    #data_mat= loadmat(data_path)


    # Carica il modello
    model, data_ = load_model('fitted_model.pkl','data_n.mat')

    # Trasforma i dati
    transformed_data = transform_data(model,data_)
    
    transform_mat = {'transformed_data': transformed_data}
    
    try:
        # Tentativo di trasformazione e salvataggio dei dati
        scipy.io.savemat(os.path.join(output_directory, 'transf_data.mat'), transform_mat)
    except Exception as e:
        print(f"Errore durante la trasformazione o il salvataggio: {e}")
    
