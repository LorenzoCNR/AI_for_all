# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:26:52 2024

@author: loren
"""

import numpy as np
from sklearn.model_selection import train_test_split
#  Function to get trial indices from multi trial exp
def get_trial_indices(trial_array, trial_numbers):
    return np.isin(trial_array, trial_numbers).nonzero()[0]
    #indices = np.isin(trial_array, trial_numbers)
    #return indices

#  shuffle trial con valori ripetuti 
def shuffle_trials(trial_array, seed=None):
    ### unique identifier per trial
    unique_trials = np.unique(trial_array)
    ## 
    if seed is not None:
        np.random.seed(seed)
    ### shuffle dei trial
    np.random.shuffle(unique_trials)
    ### mappo vecchi identificatori dei trial ai nuovi
    ### i valori finiscono nelle loro nuove posizioni
    trial_mapping = {old: new for new, old in enumerate(unique_trials)}
    shuffled_trials = np.vectorize(trial_mapping.get)(trial_array)
    return shuffled_trials


# semplice shuffle trial con valori non ripetuti
#def shuffle_trials(trials, seed=None):
 #   np.random.seed(seed)
 #   np.random.shuffle(trials)
 #   return trials

def get_trial_indices(trials, selected_trials):
    return np.isin(trials, selected_trials).nonzero()[0]

def split_data(dati, subtrain=False, shuffle=False, seed=None, train_ratio=0.7, val_ratio=0.15, verbose=False):
    """
    Splits data for each subject into training, validation, and test sets.
    Allows external specification of train, validation, and test ratios.
    
    Parameters:
    - dati: the data dictionary containing subjects
    - subtrain: whether to further split train into subtrain and validation
    - shuffle: whether to shuffle the trials
    - seed: for reproducibility in shuffling
    - train_ratio: proportion of data for training
    - val_ratio: proportion of validation data relative to test or training data
    """
    
    
# test ratio is the compl to 1
    test_ratio = 1 - train_ratio  

    for subject in dati:
        data = dati[subject]
        X = data[f"{subject}_all_0_X"]
        y = data[f"{subject}_all_0_Y"]
        trials = data[f"{subject}_all_0_Trials"]

        if shuffle:
            trials = shuffle_trials(trials, seed)
        
        unique_trials = np.unique(trials)
        
        # Split in train and test/validation
        train_trials, test_val_trials = train_test_split(unique_trials,
                                    test_size=1-train_ratio, random_state=seed)
        
        if subtrain:
            #print('we are further splitting train set')
            # Split train into subtrain and validation (within the training set)
            subtrain_trials, val_trials = train_test_split(train_trials,
                    test_size=val_ratio, random_state=seed, shuffle=False)
            test_trials = test_val_trials  # No further split for test
        else:
            #print('train set is also subtrain')
            # Further split test/validation into test and validation
            test_trials, val_trials = train_test_split(test_val_trials,
                    test_size=val_ratio, random_state=seed, shuffle=False)
            # In this case, use train as subtrain
            subtrain_trials = train_trials

        # Get indices for each set
        train_idx = get_trial_indices(trials, train_trials)
        val_idx = get_trial_indices(trials, val_trials)
        test_idx = get_trial_indices(trials, test_trials)
        subtrain_idx = get_trial_indices(trials, subtrain_trials)

        if verbose:
            print(f"subtrain_idx: {subtrain_idx}")
            print(f"train_idx: {train_idx}")
            print(f"val_idx: {val_idx}")
            print(f"test_idx: {test_idx}")

        # Assign data for subtrain (always)
        dati[subject]['X_subtrain'] = X[subtrain_idx]
        dati[subject]['y_subtrain'] = y[subtrain_idx]

        # Assign data for train, val, test
        dati[subject]['X_train'] = X[train_idx]
        dati[subject]['y_train'] = y[train_idx]
        dati[subject]['X_val'] = X[val_idx]
        dati[subject]['y_val'] = y[val_idx]
        dati[subject]['X_test'] = X[test_idx]
        dati[subject]['y_test'] = y[test_idx]

    return dati