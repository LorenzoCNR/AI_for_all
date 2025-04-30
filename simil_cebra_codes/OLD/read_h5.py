import os
os.chdir(r'C:\Users\zlollo2\Desktop\Strila_11_04_24\CNR_neuroscience\cebra_git\Cebra_for_all\cebra_codes')
import h5py
import numpy as np
from create_h5_store import create_or_open_hdf5
from create_h5_store import save_data

file_name = "rat_data_0.hdf5"
group_name = "baseline_data"
dataset_name = "rat_behav_mod"
data_neural = np.load('rat_neural.npy')
labels_std = np.load('rat_behaviour_std.npy')
labels_mod = np.load('rat_behaviour_mod.npy')

with create_or_open_hdf5(file_name) as hdf5_file:
    save_data(hdf5_file, group_name, dataset_name,labels_mod, 
              labels=labels_std, include_labels=False)


#from read_h5 import print_names


f_name='rat_data_0.hdf5'

# Just print names in the command #
def print_names(name, obj):
    print(name)
    if isinstance(obj, h5py.Group):
        print(f"{name} is a group")
    elif isinstance(obj, h5py.Dataset):
        print(f"{name} is a dataset")
    else:
        print(f"{name} Unknown type")

# 
with h5py.File(f_name, 'r') as file:
    file.visititems(print_names)
   
with h5py.File(f_name, 'r') as file:
    # Access a specific dataset
    group_= file['baseline_data']
    print(f"Content of {group_.name}:")
    for name in group_:
        print(name)


# Percorso del dataset da eliminare
dataset_path = 'baseline_data/labels'

# Apri il file HDF5 in modalità lettura/scrittura
with h5py.File(f_name, 'a') as hdf:
    # Controlla se il dataset esiste
    if dataset_path in hdf:
        # Rimuovi il dataset
        del hdf[dataset_path]
        print(f"Dataset '{dataset_path}' rimosso con successo.")
    else:
        print(f"Dataset '{dataset_path}' non trovato.")

#### sanity check
# 
with h5py.File(f_name, 'r') as file:
    file.visititems(print_names)


### ora se vogli ocaricare dei dati da un df hd5
dataset_path = '/baseline_data/rat_behav_std'

# Apri il file HDF5 in modalità lettura ('r')
with h5py.File(file_name, 'r') as hdf:
    # Controlla se il dataset esiste
    if dataset_path in hdf:
        # Acces dataset
        dataset = hdf[dataset_path]
        
        # Read from datase
        rat_behav_std_ = dataset[:]
        
        print("Succesfully loaded data!")
        print(rat_behav_std_)
    else:
        print(f"Dataset '{dataset_path}' not found!!")


### Save database structure in a txt

#output_file = 'list_groups_and_db_output.txt'

#with h5py.File(f_name, 'r') as file:
  #  with open(output_file, 'w') as out:
 #       def print_names(name, obj):
 #           line = f"{name}\n"
 #           out.write(line)
 #           if isinstance(obj, h5py.Group):
 #               out.write(f"{name} is a group\n")
  #          elif isinstance(obj, h5py.Dataset):
 #               out.write(f"{name} is a db_output\n")
 #           else:
  #              out.write(f"{name} Unkn. type\n")

 #       file.visititems(print_names)

#print(f"Names are saved in  {output_file}.")


    # Aceess  a  dataset
    #dataset = file['path/to/dataset']

    # Read a Dataset
    #data = dataset[:]
    # Get dataset shape
    #shape = dataset.shape
    
    # Get dataset data type
    # dtype = dataset.dtype
    
    # Print information
    #print(f"Shape: {shape}, Type: {dtype}")

    # Now, data contains the dataset's contents, and you can process it as needed
   # print(data)