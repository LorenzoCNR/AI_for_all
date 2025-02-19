# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 10:15:50 2025

@author: loren
"""

def adaptive_gaussian_distance(x1:torch.tensor, x2:torch.tensor)->torch.Tensor:
    """
    Calcola la distanza normalizzata usando la varianza empirica dei dati.

    Args:
        x1, x2 (torch.Tensor): Tensor di forma (batch_size, D).

    Returns:
        torch.Tensor: Matrice (batch_size, batch_size) con le distanze normalizzate.
    """
    sigma = torch.std(x1, dim=0, unbiased=True, keepdim=True)  # Deviazione standard per feature  # Calcola la deviazione standard empirica
    sigma=sigma.mean()
    dists = torch.cdist(x1, x2, p=2)
    return torch.exp(- (dists**2) / (2 * sigma**2))  # Normalizzazione Gaussiana


### generalization   
def minkowski_distance(l1, l2, p=2, sigma=1.0):
    return torch.exp(- (torch.sum(torch.abs(l1 - l2)**p, dim=-1)**(1/p)) / (2 * sigma**2))    
#### ...normalized according to variance 
def minkowski_normalized_distance(coordinates, p=2):
    # Calcolo delle deviazioni standard per normalizzazione
    std_dev = torch.std(coordinates, dim=0)
    

    # Normalize coordinates
    coordinates_normalized = coordinates / std_dev
    
    # Compute normalized minkowski distance
    return torch.cdist(coordinates_normalized, coordinates_normalized, p=p)

def time_distance(l1, l2, sigma_time):
    return torch.exp(-(l1 - l2)**2 / (2 * sigma_time**2))


def direction_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    return l1[:, None] == l2[None, :].int() 


batch=next(iter(dataloader_))

x, labels = batch  # EEG signals e labels

f_dists={'direction': direction_distance, 'position':adaptive_gaussian_distance}
labels_distance = LabelsDistance(f_dists)
label_keys = list(f_dists.keys())
multilabel=len(label_keys)>1

#### traininig
f_x=model.forward(x)
next(iter(labels))
if isinstance(labels, dict):
                if len(labels) == 1:  # Caso MONO-LABEL con dizionario
                    label_name = next(iter(labels))  # Prende l'unica chiave
                    f = (
                         f_dists[label_name]  # Prende la funzione dalla chiave
                        if isinstance(f_dists, dict)
                        else f_dists  # Usa direttamente la funzione
                    )
                    
                    
f_dists['direction']
f_dists['position']                    

labels_val[1]==labels_val[3]

if isinstance(labels, dict):
         if len(labels) == 1:  # Caso MONO-LABEL con dizionario
             label_name = next(iter(labels))  # Prende l'unica chiave
             f = (
                  f_dists[label_name]  # Prende la funzione dalla chiave
                 if isinstance(f_dists, dict)
                 else f_dists  # Usa direttamente la funzione
             )
             weight = pairwise_tensor_function(f, labels[label_name])
             return weight.view((weight.shape[0], weight.shape[1], 1))



labels_val=labels['position']
labels_val
#direction_distance(labels_val.resize(512,1), labels_val.resize(512,1))

adaptive_gaussian_distance(labels_val,labels_val)
def get_weights(f_dists, labels):
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
                         f_dists[label_name]  # Prende la funzione dalla chiave
                        if isinstance(f_dists, dict)
                        else f_dists  # Usa direttamente la funzione
                    )
                    weight = pairwise_tensor_function(f, labels[label_name])
                    return weight.view((weight.shape[0], weight.shape[1], 1))
    
                # Multi-label (più chiavi nel dizionario)
                weights = []
                for label_name, label_values in labels.items():
                    f = f_dists[label_name]  # Prende la funzione giusta dal dizionario
                    weight = pairwise_tensor_function(f, label_values)
                    weights.append(weight)
    
                return torch.stack(weights, dim=-1)  # (batch_size, batch_size, num_labels)
    
            # 🔥 Caso MONO-LABEL con funzione singola (ad es. LabelsDistance(adaptive_gaussian_distance))
        else:
                f = (
                    f_dists  # Usa direttamente la funzione
                    if callable(f_dists)
                    else next(iter(f_dists.values()))  # Caso strano in cui è un dizionario con un solo valore
                )
                weight = pairwise_tensor_function(f, labels)
                return weight.view((weight.shape[0], weight.shape[1], 1))


f_dists

pippo=get_weights(f_dists, labels)



wweight=[]
lab_dir=labels['direction']
lab_pos=labels['position']

#for label_values in lab_pos:
#print(label_values)
f = f_dists['direction'] 
f
print(f)
weight=pairwise_tensor_function(f,lab_dir)
wweight.append(weight)

wweight = torch.stack(wweight, dim=-1)



#####################################
ww=10
shift=1
    # @staticmethod
def create_windows( window, shift):
       """
       Crea finestre temporali dai dati EEG del dataset.

       Args:
           window (int): Dimensione della finestra temporale (numero di timepoints).
           shift (int): Scostamento tra finestre consecutive.

       Returns:
           DatasetEEG: Nuovo dataset contenente finestre temporali.
       """
       trials_new = []

       for trial in trials:
           signals = trial.eeg_signals
           times = trial.timepoints
           labels = trial.labels  # Dizionario delle etichette
           print(labels)
           
           return signals,times,labels

           ind_center = window // 2
           while ind_center + window // 2 < len(times):
               ind_min = ind_center - window // 2
               ind_max = ind_center + window // 2

               # Estrai finestra dai segnali
               x = signals[:, ind_min:ind_max]

               # Estrai le etichette per il centro della finestra
               window_labels = {
                      key: value[ind_center] if value.ndim == 1 else value[:, ind_center]
                      for key, value in labels.items()
                      }

               # Crea un nuovo trial con la finestra
               trials_new.append(TrialEEG(x, list(window_labels.items()), times[ind_min:ind_max]))

               # Aggiorna il centro della finestra
               ind_center += shift

       return DatasetEEG(trials_new, info=self.info)
   
sig,tim,lbs,=create_windows(ww, shift)
ind_center=ww//2
ind_min = ind_center - ww // 2
ind_max = ind_center + ww // 2
x = sig[:, ind_min:ind_max]
window_labels = {
          key: value[5] if value.ndim == 1 else value[:, 5]
          for key, value in lbs.items()
          }

trials_new.append(TrialEEG(x, list(window_labels.items()), tim[ind_min:ind_max]))

lbs.items()

list(window_labels.items())
