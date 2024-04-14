import h5py
import yaml
import joblib as jl
from pathlib import Path
import sys
#from data_h5_jl_store import create_or_open_hdf5 
### miss the warning if the databse already exists
if len(sys.argv) < 2:
    print("Too few args!!!")
def initialize_hdf5(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    hdf5_file_path = Path(config['paths']['main_path']) / config['hd5_specifics']['db_name']
    output_folder = Path(config['paths']['output_folder'])
    data_folder=Path(config['paths']['data_folder'])

    with h5py.File(hdf5_file_path, 'w') as hdf:
        # 
        data_Achilles = jl.load(data_folder / 'achilles.jl')
        hdf.create_dataset(config['hd5_specifics']['neural_data_path'], data=data_Achilles['spikes'])
        hdf.create_dataset(config['hd5_specifics']['behavior_data_path'], data=data_Achilles['position'])
        
        # Save parameters and other stuff as attributes
        for param, value in config['model_params'].items():
            hdf.attrs[param] = value
        for setting, value in config['additional_settings'].items():
            hdf.attrs[setting] = value

        # other attributes 
        hdf.attrs['model_output_path'] = str(output_folder / config['additional_settings']['model_output_path'])
        hdf.attrs['transformed_data_path'] = str(output_folder / config['additional_settings']['transformed_data_path'])
        hdf.attrs['seed'] = config['additional_settings']['seed']
        ## path for transform
        hdf.attrs['model_input_path'] = str(output_folder / config['additional_settings']['model_input_path'])
        hdf.attrs['manifold_data_path'] = config['hd5_specifics']['manifold_data_path']
        hdf.attrs['save_manifold_timestamps'] = config['hd5_specifics']['save_manifold_timestamps']


    print(f"Initialized HDF5 file at {hdf5_file_path}")
if __name__ == "__main__":
    config_path =sys.argv[1]
    initialize_hdf5(config_path)