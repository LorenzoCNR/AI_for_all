import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from typing import Optional


#### function to deal with jl data
#def load_and_inspect_jl(filepath):
 #   data = {}
 #   with open(filepath, 'r') as file:
 #       for line in file:
 #           record = json.loads(line)
 #           for key, value in record.items():
 #               if key not in data:
 #                   data[key] = []
  #              data[key].append(value)
    
 #   # convert into np array
  #  for key in data:
  #      data[key] = np.array(data[key])

  #  return data

def create_or_open_hdf5(file_name):
    #Create or open an existing db.
    return h5py.File(file_name, 'a')  # 



def save_data(hdf5_file, group_name, dataset_name, data, labels=None, include_labels=False, overwrite=False):
# Save data in an hd5 file  given specified group name and dataset
# Optionally save labels if includelabels is set on true

    # Check if group exists and (enventually) create it
    if group_name not in hdf5_file:
        group = hdf5_file.create_group(group_name)
    else:
        group = hdf5_file[group_name]
    
    # Create or overwrite data
    if dataset_name in group:
        if overwrite:
     # if dataset exists and overwrite true, remove old, create new
            print(f"Dataset '{dataset_name}' already exists and will be overwritten.")
            del group[dataset_name]
        else:
            # if overwrite  False just exit
            print(f"Dataset '{dataset_name}'already exists. No OverWrite.")
            return
    group.create_dataset(dataset_name, data=data)
    print(f"Dataset saved as '{group_name}/{dataset_name}'.")

    # Include (eventually labels)
    if include_labels and labels is not None:
        label_dataset_name = dataset_name + "_labels"
        if label_dataset_name in group:
            print(f"Labels already exist in group '{group_name}'. Skipping labels update.")
        else:
            group.create_dataset(label_dataset_name, data=labels)
     
            print(f"Labels included under '{group_name}/labels'.")


### this code store results  from model processing according to time stamp

    ##  Save manif in hdf5 file under a specified group name.
    # Optionally save labels if 'include_labels' is True and they are provided.

    # Ensure the group exists or create a new one
def save_manif(hdf5_file, group_name, manif, labels=None, include_labels=False):
    # Ensure the group exists or create a new one
    if group_name not in hdf5_file:
        group = hdf5_file.create_group(group_name)
    else:
        group = hdf5_file[group_name]

    # Generate a unique name for the dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manif_name = f"manif_{timestamp}"
    group.create_dataset(manif_name, data=manif)
    print(f"Manifold saved under '{group_name}/{manif_name}'.")

    # Optionally include labels if specified
    if include_labels and labels is not None:  # Ensure this line ends with a colon
        label_dataset_name = f"{manif_name}_labels"
        group.create_dataset(label_dataset_name, data=labels)
        print(f"Labels included under '{group_name}/{label_dataset_name}'.")

    # Create a dataset for the manif with a unique identifier based on the current timestamp


'''
def labels_to_str(labels):
    # Convert labels in a numeric string descriptive
    labels_str = "_".join([str(int(label)) for label in np.unique(labels[:, 1])])
    retur'n "labels_" + labels_str
  

def generate_group_name(labels):
    
    class_id = labels[0, 1]  
    return f"class_{int(class_id)}"
    '''

def save_transformed_data(output_folder, data, file_name='transformed_data.hdf5'):
    """

"""

    save_manif(hdf5_file, group_name, data, labels=labels, include_labels=include_labels)
    print(f"Transformed data and labels saved in {group_name}")




def save_fig_with_timestamp(fig, fig_id, IMAGES_PATH, tight_layout=True, fig_extension="png", resolution=300):
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plot name according to time stamp
    filename = f"{fig_id}_{timestamp}.{fig_extension}"
    # 
    path = IMAGES_PATH / filename
    # Apply tight layout  if requested
    if tight_layout:
        plt.tight_layout()
    # Save image
    fig.savefig(path, format=fig_extension, dpi=resolution)
    plt.close(fig)


#### create and/or modify a database
'''
def create_or_modify_hdf5(file_name):

    with h5py.File(file_name, 'a') as hdf:
        
        if 'gruppo1' not in hdf:
            gruppo1 = hdf.create_group('gruppo1')
        else:
            gruppo1 = hdf['gruppo1']
        

        data1 = np.random.randn(100)  
        if 'dataset1' in gruppo1:
            del gruppo1['dataset1']  
        dataset1 = gruppo1.create_dataset('dataset1', data=data1)
        
        
        if 'gruppo2' not in hdf:
            gruppo2 = hdf.create_group('gruppo2')
            data2 = np.random.randn(200) 
            dataset2 = gruppo2.create_dataset('dataset2', data=data2)
'''
####


#### paramters storing

def save_parameters(hdf5_file, group_name, parameters):
    """Salva i parametri nel file HDF5."""
    if group_name not in hdf5_file:
        group = hdf5_file.create_group(group_name)
    else:
        group = hdf5_file[group_name]

    for key, value in parameters.items():
        group.attrs[key] = value

    print(f"Parameters saved under group '{group_name}'.")
