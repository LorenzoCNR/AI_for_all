"""
    Main per codice davide
    Cambiare la main root in riga 25 e dintorni

    Esempio di chiamata da terminale:
    python nome_script.py --input_dir "input_path" --output_dir "output_path" --name "achilles" 
    --filters 32 --tau 0.1 --epochs 250 --dropout 0.5 --latents 3 --window 10 --sigma_pos 0.016 
    --sigma_time 0.025
    
    Se non si specificano i parametri, verranno utilizzati i valori di default definiti da argparse.
     
    DA CAMBIARE:
        aggiungere al parse i vicini 'n_n' del knn decoder e la metrica (cosine)
    
    
    
    """

##possiamo aggiungere al parser altre cose...tipo il sampling rate, lo shift, il  et al.

import os
import sys
from pathlib import Path
import argparse
i_dir=r'/media/zlollo/21DB-AB79/AI_PhD_Neuro_CNR/Empirics/GIT_stuff/AI_for_all/Contrastive_Stuff/'
i_dir=r'J:\AI_PhD_Neuro_CNR\Empirics\GIT_stuff\AI_for_all\Contrastive_Stuff'
os.chdir(i_dir)
def setup_paths():
    """
   Dynamic path config
    """
   # If __file__ is not available, use the current working directory (cwd).
    try:
        project_root = Path(__file__).resolve().parent
    except NameError:
        project_root = Path(os.getcwd())  # Fallback to cwd if __file__ is not available
    

    print(f"Project root is: {project_root}")
    # Percorso a "EEG-ANN-Pipeline"
    eeg_pipeline_path = project_root / "EEG-ANN-Pipeline"
    print(f"Oath to eeg-ann-pipeline': {eeg_pipeline_path}")
    default_input_dir= project_root.parent/ "data" / "rat_hippocampus"
    print(f"data_path: {default_input_dir}")
    # Percorso alla directory "contrastive_output" (output directory)
    default_output_dir = project_root / "contrastive_output"
    print(f"ouptut path: {default_output_dir}")

    # Aggiungi "EEG-ANN-Pipeline" al path
    if str(eeg_pipeline_path) not in sys.path:
        sys.path.append(str(eeg_pipeline_path))

    # Crea la directory di output se non esiste
    default_output_dir.mkdir(exist_ok=True)

    # Cambia la directory di lavoro alla radice del progetto
    #os.chdir(project_root)

    # Ritorna i percorsi configurati
    return project_root, eeg_pipeline_path, default_output_dir, default_input_dir




#project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths()
#from d_code_cnn1 import run_d_code  # Importa lo script principale



###WINDOWS
#main_root = r"F:\........"
#input_dir=r"F:\............"
### UBUNTU
#main_root = r"/media/............."

#input_dir=r"/media/................."


#print(sys.path)
   # Check paths
def main(args):
   
    z_train, z_val, labels_train, labels_val, posdir_decode_CL = run_d_code(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        name=args.name,
        filters=args.filters,
        tau=args.tau,
        epochs=args.epochs,
        dropout=args.dropout,
        latents=args.latents,
        ww=args.ww,
        sigma_pos=args.sigma_pos,
        sigma_time=args.sigma_time,
        train_split=args.train_split,
        valid_split=args.valid_split,
        l_rate=args.l_rate,
        batch_size=args.batch_size,
        fs=args.fs,
        shift=args.shift,
        normalize=args.normalize,
        neighbors=args.neighbors
    )

    return z_train, z_val, labels_train, labels_val, posdir_decode_CL

if __name__ == "__main__":
  
        
    project_root, eeg_pipeline_path, default_output_dir, default_input_dir = setup_paths()
    from d_code_cnn1 import run_d_code  

 
    parser = argparse.ArgumentParser(description="Execute the Davide pipeline.")
    # Default value for --name
    parser.add_argument("--input_dir", type=str, default=str(default_input_dir), help="Input directory")
    parser.add_argument("--output_dir", type=str,default=str(default_output_dir), help="Output directory")
    parser.add_argument("--name", type=str, default="achilles", help="Name of the subject (default: 'subject1')")
    parser.add_argument("--filters", type=int, default=32, help="Number of filters in the model")
    parser.add_argument("--tau", type=float, default=0.1, help="Temperature for contrastive loss")
    parser.add_argument("--epochs", type=int, default=250, help="Number of training epochs")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate in the model")
    parser.add_argument("--latents", type=int, default=3, help="Dimension of the latent space")
    parser.add_argument("--ww", type=int, default=10, help="Window size in time points")
    parser.add_argument("--sigma_pos", type=float, default=0.016, help="Sigma for position label distance")
    parser.add_argument("--sigma_time", type=float, default=0.025, help="Sigma for time label distance")
    parser.add_argument("--train_split", type=float, default=0.85, help="Proportion of training data")
    parser.add_argument("--valid_split", type=float, default=0.15, help="Proportion of validation data")
    parser.add_argument("--l_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--fs", type=int, default=40, help="Sampling frequency (Hz)")
    parser.add_argument("--shift", type=int, default=1, help="Shift in time points for sliding windows")
    parser.add_argument("--normalize", type=bool, default=True, help="Apply normalization in the model")
    parser.add_argument("--neighbors", type=int, default=25, help="Number of neighbors for knn")


    args = parser.parse_args()

    args.name='achilles'
    args.epochs=250
    args.tau=1
    args.filters=32
    #args.sigma_time=0.5
    #args.sigma_pos=0.5
    z_train, z_val ,labels_train, labels_val,posdir_decode_CL=main(args)
    
'''
     #parser.add_argument("--n_iter", type=int, default=10, help="Number of random search iterations")
     #         n_iter=args.n_iter,

    parser.add_argument("--tau_bounds", type=float, nargs=2, default=[0.01, 0.2], help="Bounds for tau (temperature)")
    parser.add_argument("--filters_bounds", type=int, nargs=2, default=[16, 128], help="Bounds for number of filters")
    parser.add_argument("--dropout_bounds", type=float, nargs=2, default=[0.1, 0.5], help="Bounds for dropout rate")
    parser.add_argument("--neighbors_bounds", type=int, nargs=2, default=[5, 50], help="Bounds for KNN neighbors")
    from scipy.stats import randint, uniform
    import pandas as pd
    
    param_distribs = {
        'tau': uniform(args.tau_bounds[0], args.tau_bounds[1] - args.tau_bounds[0]),
        'filters': randint(args.filters_bounds[0], args.filters_bounds[1]),
        'dropout': uniform(args.dropout_bounds[0], args.dropout_bounds[1] - args.dropout_bounds[0]),
        'neighbors': randint(args.neighbors_bounds[0], args.neighbors_bounds[1]),
    }


    results = []

    # Random Search Loop
    for i in range(args.n_iter):
        # Estrai valori casuali dai bounds
        tau = param_distribs['tau'].rvs()
        filters = param_distribs['filters'].rvs()
        dropout = param_distribs['dropout'].rvs()
        neighbors = param_distribs['neighbors'].rvs()
        
        print(f"Iteration {i+1}/{args.n_iter}: tau={tau:.4f}, filters={filters}, dropout={dropout:.4f}, neighbors={neighbors}")
        
        #update args with randomized parmas 
        args.tau = tau
        args.filters = filters
        args.dropout = dropout
        
       
        z_train, z_val, labels_train, labels_val, posdir_decode_CL = main(args)
        
        #  Save Results
        result = {
            'iteration': i + 1,
            'tau': tau,
            'filters': filters,
            'dropout': dropout,
            'neighbors': neighbors,
            'test_score': posdir_decode_CL[0],
            'pos_test_error': posdir_decode_CL[1],
            'pos_test_score': posdir_decode_CL[2],
        }
        results.append(result)

    # Save to csv
    df_results = pd.DataFrame(results)


     # find best result
    best_result = min(results, key=lambda x: x['pos_test_error'])
    print("Best result:", best_result)
   
    '''

