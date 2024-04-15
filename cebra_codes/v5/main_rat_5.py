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
    "seed": "32",
    "InNeural": "neural_data.hd5",
    "InBehavior": "behavior_data.hd5"
    }

#from hip_models_1 import run_hip_models

output_filename = "cebra_manifold_5.hdf5"
model_filename  = "cebra_model.pkl"  
def main(params):
    #create_database() 
    random.seed(params['seed']);
    maxI = 10^5;
    N = 1
    group_name= 'data'
    ### create a group of manifold per label
           
    for n in range(1, N+1):
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
        
        # get params
        mod_arch=params.get("model_architecture")
        out_dim=int(params.get("output_dimension"))
        temp=int(params.get("temperature"))
        max_iter=int(params.get("max_iterations"))
        dist=params.get("distance")
        cond=params.get("conditional")
        time_off=int(params.get("time_offsets"))
        hyb=params.get("hybrid").strip('"')
        batch_s=int(params.get("batch_size"))
        l_r=float(params.get("learning_rate"))
        
        
        
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
        with h5py.File(output_filename, 'w') as out_hdf:
            group = out_hdf.create_group(group_name)
            group.create_dataset("manifold", data=manif,        compression='gzip', compression_opts=9)
            group.create_dataset("behavior",   data=behavior_data,compression='gzip', compression_opts=9)
            print(f"Transformed data saved in {output_filename}")
        
    return manif, behavior_data
    
if __name__=="__main__":
    main(params)
        
try:
    with h5py.File(output_filename, 'r') as f:
        print(list(f.keys()))  # Stampa l'elenco dei gruppi/dataset per verificare la struttura
except Exception as e:
    print(e)
