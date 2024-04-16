#!/usr/bin/env python3

import sys
import matplotlib
import h5py
matplotlib.use('TkAgg')
import numpy as np
import random
import math
import torch
import cebra.datasets
import joblib as jl
from cebra import CEBRA

manifold_filename   = "manifold.hdf5"
manifold_field      = "manifold"
maxI                = 10^5;
        
    
def main(modelParams_filename):
    #modelParams
    with h5py.File(modelParams_filename, 'r') as hdf:
        params = {
            'model_architecture': hdf.attrs['model_architecture'],
            'batch_size': int(hdf.attrs['batch_size']),
            'learning_rate': float(hdf.attrs['learning_rate']),
            'temperature': int(hdf.attrs['temperature']),
            'output_dimension': int(hdf.attrs['output_dimension']),
            'max_iterations': int(hdf.attrs['max_iterations']),
            'distance': hdf.attrs['distance'],
            'conditional': hdf.attrs['conditional'],
            'time_offsets': int(hdf.attrs['time_offsets']),
            'hybrid': hdf.attrs['hybrid'],
            'verbose': hdf.attrs['verbose']
        }
        group_field         = hdf.attrs['group_field']
        behavior_field      = hdf.attrs['behavior_field']
        neural_field        = hdf.attrs['neural_field']
        model_filename      = hdf.attrs['model_filename']
        neural_filename     = hdf.attrs['neural_filename']
        behavior_filename   = hdf.attrs['behavior_filename']
        seed                = hdf.attrs['seed']

    # seed 
    random.seed(seed)                                   # random
    np.random.seed(random.randint(1,maxI))              # numpy
    # seed PyTorch
    torch.manual_seed(random.randint(1,maxI))
    torch.cuda.manual_seed_all(random.randint(1,maxI))  # multi-GPU
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic  = True      # may reduce performance
        torch.backends.cudnn.benchmark      = False

    # get dataset
    hippocampus_pos = cebra.datasets.init('rat-hippocampus-single-achilles')
    neural_data     = hippocampus_pos.neural
    behavior_data   = hippocampus_pos.continuous_index.numpy()
    
    # save neural and behavior in hd5
    with h5py.File(behavior_filename, 'w') as out_hdf:
        group = out_hdf.create_group(group_field)
        group.create_dataset(behavior_field, data=behavior_data,compression='gzip', compression_opts=9)
        print(f"{behavior_field} saved in {behavior_filename}")
    with h5py.File(neural_filename, 'w') as out_hdf:
        group = out_hdf.create_group(group_field)
        group.create_dataset(neural_field, data=neural_data,compression='gzip', compression_opts=9)
        print(f"{neural_field} saved in {neural_filename}")
 
    # get params
    mod_arch    =params.get("model_architecture")
    out_dim     =int(params.get("output_dimension"))
    temp        =int(params.get("temperature"))
    max_iter    =int(params.get("max_iterations"))
    dist        =params.get("distance")
    cond        =params.get("conditional")
    time_off    =int(params.get("time_offsets"))
    hyb         =params.get("hybrid").strip('"')
    batch_s     =int(params.get("batch_size"))
    l_r         =float(params.get("learning_rate"))

    # load behavior
    with h5py.File(behavior_filename, 'r') as hdf:
        behavior_data  = hdf[group_field + '/' +  behavior_field][:]
    with h5py.File(neural_filename, 'r') as hdf:
        neural_data    = hdf[group_field + '/' +  neural_field][:]


    cebra_model = CEBRA(model_architecture=mod_arch,
                        batch_size=batch_s,
                        learning_rate=l_r,
                        temperature=temp, 
                        output_dimension=out_dim,
                        max_iterations=max_iter,
                        distance=dist,
                        conditional=cond,
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=time_off,
                        hybrid=hyb)

    #cebra_model = CEBRA(**params)

    cebra_model.fit(neural_data,behavior_data)
    manif       = cebra_model.transform(neural_data)
    
    jl.dump(cebra_model, model_filename)
    print(f"Model saved at {model_filename}")
    cebra_model  = jl.load(model_filename)
    
    # save data in 
    with h5py.File(manifold_filename, 'w') as out_hdf:
        group = out_hdf.create_group(group_field)
        group.create_dataset(manifold_field, data=manif,        compression='gzip', compression_opts=9)
        print(f"{manifold_field} saved in {manifold_filename}")   
    return
    
if __name__=="__main__":
    modelParams_filename = sys.argv[1]
    main(modelParams_filename)
        
#try:
#    with h5py.File(manifold_filename, 'r') as f:
#        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
#except Exception as e:
#    print(e)
