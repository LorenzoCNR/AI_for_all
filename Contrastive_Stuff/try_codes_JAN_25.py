# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:43:32 2025

@author: loren
"""
import timeit
print(timeit.timeit('{}'))  # Output: molto veloce
print(timeit.timeit('dict()'))  #

num_tps=dataset_windows.num_timepoints
num_chn=dataset_windows.num_channels
num_trs=dataset_windows.num_trials
dataset_windows.labels_type  
dataset_windows.labels_format


dataset_windows.labels_format['position']
# Creo un tensore per tenere tutti i dati
eeg_signals = torch.zeros((num_trs, 1,num_chn,num_tps), dtype=torch.float32)
eeg_signals.shape
dataset_windows.eeg_signals=eeg_signals
for i, trial in enumerate(dataset_windows.trials):
            eeg_signals[i, 0, :, :] = torch.from_numpy(trial.eeg_signals)

        # Caso singola label: creo un tensore  (se sono stringhe devo prima convertirle in interi)
        # Caso multi-label: creo un dizionario e a ogni tipo di label associo un tensore
create_labels(selected_labels)

def __len__(dataset):
        return dataset.num_trials

def __getitem__(dataset, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = dataset.eeg_signals[idx, :, :, :]

        # Le label dipendono dal caso
        if dataset.labels_type == 'single_label':
            y = dataset.labels[idx]
        else:
            y = {label_name: dataset.labels[label_name][idx] for label_name in dataset.label_names }

        return x, y
 

labels = [trial.labels['position'] for trial in dataset_windows.trials]
labels_list = [list(trial.labels.values())[0] for trial in dataset_windows.trials]
labels = [list(trial.labels.values())[0] for trial in dataset_windows.trials]

labels = [trial.labels for trial in dataset.trials]


label_names = list(dataset_windows.trials[0].labels.keys())
label_names=[label_name for label_name in dataset_windows.labels_format]

dataset_windows.trials[0].labels
    
 
len_d=__len__(dataset_windows)

item_d=__getitem__(dataset_windows,43)

selected_labels=['position', 'direction']

for l_name in label_names:
    print(l_name)
    labels_i = [trial.labels[l_name] for trial in dataset_windows.trials]
    print(l_name, labels_i)
                # Controllo se sono stringhe e nel caso le converto
                if self.labels_format[label_name] == 'string':

                    labels_i, labels_int_to_str_i = convert_labels_string_to_int(labels_i)
                    labels_int_to_str[label_name] = labels_int_to_str_i

                # Converto in base anche al tipo di dato
                if self.labels_format[label_name] == 'float':
                    labels_i = torch.FloatTensor(labels_i)
                else:
                    labels_i = torch.LongTensor(labels_i)

                # Le salvo nel dizionario globale
                labels[label_name] = labels_i
            
            # Salvo le label nella classe
            self.labels = labels

            # Salvo il dizionario di conversione se non è vuoto
            if len(labels_int_to_str) > 0:
                self.labels_int_to_str = labels_int_to_str            

# ---------------------------- My modified InfoNCE --------------------------- #
import torch
import torch
import torch


label_distance={}

class LabelsDistance:
    def __init__(self, labels_distance_functions):
        """
        Inizializza LabelsDistance.

        Args:
            labels_distance_functions (dict o funzione singola):
                - Se è un dizionario → multi-label (es. posizione + direzione).
                - Se è una funzione singola → mono-label.
        """
        if isinstance(labels_distance_functions, dict):
            self.labels_distance_functions = labels_distance_functions
            self.label_keys = list(labels_distance_functions.keys())  # Lista delle chiavi disponibili
            self.multi_label = len(self.label_keys) > 1  # Multi-label se ci sono più chiavi
        else:
            self.labels_distance_functions = labels_distance_functions  # Assegna direttamente la funzione
            self.label_keys = None  # Nessuna chiave perché non è un dizionario
            self.multi_label = False  # Non è multi-label

    def get_weights(self, labels):
        """
        Calcola la matrice di distanze tra le etichette nel batch.

        Args:
            labels (dict o tensore singolo): 
                - Se multi-label, è un dizionario (es. {"position": tensor(512,2), "direction": tensor(512,)}).
                - Se mono-label, può essere un tensore singolo o un dizionario con una sola chiave.

        Returns:
            torch.Tensor: Matrice di distanze (batch_size, batch_size, num_labels) se multi-label,
                          oppure (batch_size, batch_size, 1) se mono-label.
        """
        
        def pairwise_tensor_function(f, x):
            return f(x, x.T) if x.ndim == 1 else f(x, x)

        # 🔥 Se labels è un dizionario (MULTI-LABEL o MONO-LABEL con una sola chiave)
        if isinstance(labels, dict):
            if len(labels) == 1:  # Caso MONO-LABEL con dizionario
                label_name = next(iter(labels))  # Prende l'unica chiave
                f = (
                    self.labels_distance_functions[label_name]  # Prende la funzione dalla chiave
                    if isinstance(self.labels_distance_functions, dict)
                    else self.labels_distance_functions  # Usa direttamente la funzione
                )
                weight = pairwise_tensor_function(f, labels[label_name])
                return weight.view((weight.shape[0], weight.shape[1], 1))

            # Multi-label (più chiavi nel dizionario)
            weights = []
            for label_name, label_values in labels.items():
                f = self.labels_distance_functions[label_name]  # Prende la funzione giusta dal dizionario
                weight = pairwise_tensor_function(f, label_values)
                weights.append(weight)

            return torch.stack(weights, dim=-1)  # (batch_size, batch_size, num_labels)

        # 🔥 Caso MONO-LABEL con funzione singola (ad es. LabelsDistance(adaptive_gaussian_distance))
        else:
            f = (
                self.labels_distance_functions  # Usa direttamente la funzione
                if callable(self.labels_distance_functions)
                else next(iter(self.labels_distance_functions.values()))  # Caso strano in cui è un dizionario con un solo valore
            )
            weight = pairwise_tensor_function(f, labels)
            return weight.view((weight.shape[0], weight.shape[1], 1))
    
    
    