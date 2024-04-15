#!/usr/bin/env python3

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

#main_path=r'/media/zlollo/STRILA/CNR_neuroscience/cebra_git/Cebra_for_all/cebra_codes'
main_path=r'/home/donnarumma/tools/Cebra_for_all/cebra_codes'

params = {
    "model_architecture": 'offset10-model',
    "batch_size": "512",
    "learning_rate": "3e-4",
    "temperature": "1",
    "output_dimension": "3",
    "max_iterations": "10000",
    "distance": "cosine",
    "conditional": "time_delta",
    "hybrid": "False",
    "time_offsets": "10",
    "seed": "32"
    }

#from hip_models_1 import run_hip_models

neural_filename     = "neural.hd5"
manifold_filename   = "manifold.hdf5"
behavior_filename   = "behavior.hdf5"
model_filename      = "cebra_model.pkl"
manifold_field      = "manifold"
behavior_field      = "behavior"
neural_field        = "neural"
group_field         = "data"       
    
def main(params):
    #create_database() neural
    random.seed(params['seed']);
    maxI = 10^5;
    # seed
    params['seed']  =random.randint(1,maxI)
    sd = params.get("seed");
    random.seed(sd)
    np.random.seed(int(abs(math.log(random.random()))))
    # Imposta seed per PyTorch
    torch.manual_seed(int(abs(math.log(random.random()))))
    torch.cuda.manual_seed_all(int(abs(math.log(random.random()))))  # Per multi-GPU
    torch.backends.cudnn.deterministic = True  # Potrebbe ridurre le prestazioni
    torch.backends.cudnn.benchmark = False

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

    #isequalneural       = np.array_equal(neural_data2,neural_data)
    #print(f'neural equal: {isequalneural}')
    #isequalbehavior     = np.array_equal(behavior_data2,behavior_data)
    #print(f'behavior equal: {isequalbehavior}')
    #input("Wait")

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
    main(params)
        
#try:
#    with h5py.File(manifold_filename, 'r') as f:
#        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
#except Exception as e:
#    print(e)
