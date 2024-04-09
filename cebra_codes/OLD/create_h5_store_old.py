import h5py
import numpy as np
from datetime import datetime

def create_or_open_hdf5(file_name):
    #Create or open an existing db.
    #'a' open in r/w and create file (while not existing)
    return h5py.File(file_name, 'a')  # 

def save_manif(hdf5_file, group_name, manif, labels=None, include_labels=False):
    ##  Save manif in hdf5 file under a specified group name.
    # Optionally save labels if 'include_labels' is True and they are provided.

    # Check if the group exists, if not, create it
    if group_name not in hdf5_file:
        group = hdf5_file.create_group(group_name)
    else:
        group = hdf5_file[group_name]
    
    # Optionally include labels if specified
    if include_labels and labels is not None:
        # You can decide to overwrite existing labels or skip if they already exist
        if "labels" in group:
            print(f"Labels already exist in group '{group_name}'. Skipping labels update.")
        else:
            group.create_dataset("labels", data=labels)
            print(f"Labels included under '{group_name}/labels'.")

    # Create a dataset for the manif with a unique identifier based on the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manif_name = f"manif_{timestamp}"
    group.create_dataset(manif_name, data=manif)
    print(f"Manifold saved under '{group_name}/{manif_name}'.")

'''
def labels_to_str(labels):
    # Convert labels in a numeric string descriptive
    labels_str = "_".join([str(int(label)) for label in np.unique(labels[:, 1])])
    retur'n "labels_" + labels_str
  

def generate_group_name(labels):
    
    class_id = labels[0, 1]  
    return f"class_{int(class_id)}"
    '''