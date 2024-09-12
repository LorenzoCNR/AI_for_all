 ### update tutto in un file h5   
#import h5f
import h5py
import os
import yaml
import numpy as np
import joblib
import re
import torch
from io import BytesIO
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
import sklearn.metrics
################################# Hd5 FILES MANAGEMENT ###################################
## save model and parameters in hd5 (con torch)
def save_model_to_hdf5(model, model_params, hdf5_path, group_name,error):
    """
    Salva un modello e i suoi parametri in un file HDF5.
    - `model`: modello da salvare.
    - `model_params`: parametri del modello.
    - `hdf5_path`: percorso del file HDF5.
    - `group_name`: nome del gruppo in cui salvare il modello.
    """
    #with BytesIO() as buffer:
    model_path = f"{group_name}_model.pth"
    torch.save(model, model_path)       #buffer.seek(0)

    # model serialization on temporary buffer
    #buffer = BytesIO()
   # joblib.dump(model, buffer)
### buffer punctator to start
       #buffer.seek(0)
     
    with h5py.File(hdf5_path, 'a') as h5f:
        # If group exists, overwrite it
        if group_name in h5f:
            del h5f[group_name]
        grp = h5f.create_group(group_name)

        # Save model params as group attributes
        for key, value in model_params.items():
            grp.attrs[key] = value
         # Salva l'errore
        grp.attrs['best_error'] = error

        # # Salva il modello serializzato nel dataset 'model_data'
       # grp.create_dataset('model_data', data=np.array(buffer.getvalue()))
        grp.attrs['pytorch_model_path'] = model_path
        #buffer.close()

    print(f"Data saved under {group_name} in {hdf5_path}")
    return hdf5_path, group_name  

##### Iterate and explore h5 file
#os.getcwd()
### try woth achilles_results
#f_name='achilles_models.hdf5'
#group_name = 'best_model_knn'
#f_name='achille_cebra_time_model.hdf5'


 ### function to explore  h5 files
def print_names(name, obj):
    print(name)
    if isinstance(obj, h5py.Group):
        print(f"{name} is a group")
        # Aggiungi la stampa degli attributi del gruppo
        for key, value in obj.attrs.items():
            print(f"  Attribute {key}: {value}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{name} is a dataset")
        # Aggiungi la stampa degli attributi del dataset
        for key, value in obj.attrs.items():
            print(f"  Attribute {key}: {value}")
    else:
        print(f"{name} Unknown type")

#with h5py.File(f_name, 'r') as file:
#    file.visititems(print_names)

def load_model_from_hdf5(hdf5_path, group_name):
    with h5py.File(hdf5_path, 'r') as file:
        # hec group exists
        if group_name in file:
            grp = file[group_name]
            
            # load model parameters
            model_params = {key: value for key, value in grp.attrs.items()}
            
            # read binary datas
            #model_data = grp['model_data'][()]
            
            # Load best (min) error
            best_error = grp.attrs['best_error']
            # Carica il percorso del file del modello PyTorch
            model_path = grp.attrs['pytorch_model_path']
       
            # Carica il modello PyTorch
            model = torch.load(model_path)

            
            # Use buffer to load model
            #buffer = BytesIO(model_data)
            # Inizializza il modello e carica i pesi dallo state dict
            #model = model_class()  # Devi passare la classe del modello
            #model.load_state_dict(torch.load(buffer))
           # model = joblib.load(buffer)

            return model, model_params, best_error
        else:
            raise ValueError(f"Group {group_name} not found in {hdf5_path}")


#try:
#    model_knn, model_params_knn = load_model_from_hdf5(f_name, group_name)
#    print("Model loaded successfully!")
#    print("Model parameters:", model_params_knn)
#except ValueError as e:
#    print(e)



def rename_datasets_and_update_paths(h5_file):
    with h5py.File(h5_file, "a") as h5f:
        for rat in list(h5f.keys()):
            rat_group = h5f[rat]
            new_rat_name = re.sub(r'[0-9]', '', rat)
            if new_rat_name != rat:
                h5f.move(rat, new_rat_name)
                rat_group = h5f[new_rat_name]
                print(f"Renamed group '{rat}' to '{new_rat_name}'")
            
            for model_type in list(rat_group.keys()):
                model_group = rat_group[model_type]
                new_model_name = re.sub(r'[0-9]', '', model_type)
                if new_model_name != model_type:
                    rat_group.move(model_type, new_model_name)
                    model_group = rat_group[new_model_name]
                    print(f"Renamed subgroup '{model_type}' to '{new_model_name}'")
                
                for dataset_name in list(model_group.keys()):
                    new_dataset_name = re.sub(r'[0-9]', '', dataset_name)
                    new_dataset_name = new_dataset_name.replace('_model', '')
                    if new_dataset_name != dataset_name:
                        print(f"Renaming '{dataset_name}' to '{new_dataset_name}'")
                        rename_dataset(h5f, f"{new_rat_name}/{new_model_name}", dataset_name, new_dataset_name)
                
                # Aggiorna i percorsi nei path degli state dict e dei modelli
                if 'state_dict_path' in model_group.attrs:
                    old_path = model_group.attrs['state_dict_path']
                    new_path = re.sub(r"_[0-9]+.*.pth", ".pth", old_path)
                    model_group.attrs['state_dict_path'] = new_path
                    print(f"Updated state dict path from '{old_path}' to '{new_path}'")

                if 'model_path_full' in model_group.attrs:
                    old_path = model_group.attrs['model_path_full']
                    new_path = re.sub(r"_[0-9]+.*.pth", ".pth", old_path)
                    model_group.attrs['model_path_full'] = new_path
                    print(f"Updated model path from '{old_path}' to '{new_path}'")

def rename_dataset(hdf_file, group_name, old_name, new_name):
    group = hdf_file[group_name]
    
    if old_name in group:
        if new_name in group:
            print(f"Skipping: Dataset '{new_name}' already exists in group '{group_name}'.")
            return False  
            
        old_dataset = group[old_name]
        data = np.array(old_dataset)
        
        # Create new dataset with new name
        group.create_dataset(new_name, data=data)
        
        # Copy attributes from existing dataset
        for attr_name, attr_value in old_dataset.attrs.items():
            group[new_name].attrs[attr_name] = attr_value
        
        # Delete original dataset
        del group[old_name]
        
        print(f"Dataset '{old_name}' renamed to '{new_name}' in group '{group_name}'")
        return True
    else:
        print(f"Dataset '{old_name}' not found in group '{group_name}'")
        return False

def get_dataset_names(file_path):
    with h5py.File(file_path, "r") as h5f:
        groups = list(h5f.keys())
        dataset_names = {group: list(h5f[group].keys()) for group in groups}
    return dataset_names       

def verify_results(h5_file):
    with h5py.File(h5_file, "r") as h5f:
        for rat in h5f.keys():
            print(f"{rat} is a group")
            rat_group = h5f[rat]
            for model_type in rat_group.keys():
                print(f"{rat}/{model_type} is a group")
                model_group = rat_group[model_type]
                for dataset_name in model_group.keys():
                    print(f"{rat}/{model_type}/{dataset_name} is a dataset")
                    dataset = model_group[dataset_name]
                    print(f"Parameters: {dataset.attrs['params']}")
                
                # Verifica gli attributi dei percorsi del modello e dello state dict
                if 'state_dict_path' in model_group.attrs:
                    print(f"State dict path: {model_group.attrs['state_dict_path']}")
                if 'model_path_full' in model_group.attrs:
                    print(f"Full model path: {model_group.attrs['model_path_full']}")
# Path to your HDF5 file



def save_results(h5_file, rat, model_type, params, model, embeddings, neural_data, behavioral_data, output_dir, replace=False, is_grid_search=False):
    param_name = '_'.join(f"{k}{v}" for k, v in sorted(params.items())) if is_grid_search else ""
    
    with h5py.File(h5_file, "a") as h5f:
        # Group for rat
        if rat not in h5f:
            rat_group = h5f.create_group(rat)
        else:
            rat_group = h5f[rat]
        
        # Subgroup for model
        if model_type not in rat_group:
            model_group = rat_group.create_group(model_type)
        else:
            model_group = rat_group[model_type]
        
        # Save embedding
        embedding_name = f"{model_type}_embedding"
        if is_grid_search:
            embedding_name = f"{model_type}_embedding_{param_name}"
        
        if embedding_name in model_group:
            if replace:
                del model_group[embedding_name]
                dataset = model_group.create_dataset(embedding_name, data=embeddings)
                dataset.attrs['params'] = yaml.dump(params)
                print(f"Dataset {embedding_name} replaced.")
            else:
                print(f"Dataset {embedding_name} already exists, skipping save.")
        else:
            dataset = model_group.create_dataset(embedding_name, data=embeddings)
            dataset.attrs['params'] = yaml.dump(params)
            print(f"Dataset {embedding_name} saved.")
        
        # Save behavioral and neural data (only once per rat)
        original_data_group_name = "original_data"
        if original_data_group_name not in rat_group:
            original_data_group = rat_group.create_group(original_data_group_name)
            original_data_group.create_dataset("neural_data", data=neural_data)
            original_data_group.create_dataset("behavioral_data", data=behavioral_data)
            print(f"Original data for {rat} saved.")
        
        # Save model and state dict
        model_file_name = f"{rat}_{model_type}_model.pth"
        state_dict_file_name = f"{rat}_{model_type}_state_dict.pth"
        
        if is_grid_search:
            model_file_name = f"{rat}_{model_type}_model_{param_name}.pth"
            state_dict_file_name = f"{rat}_{model_type}_state_dict_{param_name}.pth"
        
        model_path = os.path.join(output_dir, model_file_name)
        state_dict_path = os.path.join(output_dir, state_dict_file_name)
        
        torch.save(model, model_path)
        torch.save(model.state_dict(), state_dict_path)
        
        model_group.attrs['model_path'] = model_path
        model_group.attrs['state_dict_path'] = state_dict_path
        
        print(f"Model {model_file_name} saved.")
        print(f"State dict {state_dict_file_name} saved.")

