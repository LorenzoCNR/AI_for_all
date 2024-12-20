import torch
import numpy as np
from data import DatasetEEG
from torch.utils.data import Dataset
from helpers.eeg_utils import convert_labels_string_to_int


class DatasetEEGTorch(Dataset):

    def __init__(self, dataset: DatasetEEG):
        
        # Controllo cohe i trials siano tutti della stessa lunghezza
        # altrimenti non posso creare il dataset pytorch
        if not isinstance(dataset.num_timepoints, int):
            raise ValueError('I trial non hanno tutti la stessa lunghezza')
        
        # Mantengo un riferimento al dataset da cui previene
        # e altre informazioni utili
        self.dataset_original = dataset
        self.labels_type = dataset.labels_type
        self.labels_format = dataset.labels_format

        # Per comodità salvo anche qui come attributi le info sul dataset
        self.num_trials = dataset.num_trials
        self.num_timepoints = dataset.num_timepoints
        self.num_channels = dataset.num_channels
        
        # Creo un tensore per tenere tutti i dati
        self.eeg_signals = torch.zeros((self.num_trials, 1, self.num_channels, self.num_timepoints), dtype=torch.float32)

        for i, trial in enumerate(dataset.trials):
            self.eeg_signals[i, 0, :, :] = torch.from_numpy(trial.eeg_signals)

        # Caso singola label: creo un tensore  (se sono stringhe devo prima convertirle in interi)
        # Caso multi-label: creo un dizionario e a ogni tipo di label associo un tensore
        self.create_labels()

    def __len__(self):
        return self.num_trials

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.eeg_signals[idx, :, :, :]

        # Le label dipendono dal caso
        if self.labels_type == 'single_label':
            y = self.labels[idx]
        else:
            y = {label_name: self.labels[label_name][idx] for label_name in self.label_names }

        return x, y
    
    def create_labels(self):

        # Separo il caso single-label da quello multi-label

        # Caso single label
        if self.labels_type == 'single_label':

            # Creo una lista con le label
            labels = [trial.label for trial in self.dataset_original.trials]

            # Controllo se sono stringhe
            if self.labels_format == 'string':

                # Nel caso le converto e salvo il dizionario per tornare indietro
                labels, labels_int_to_str = convert_labels_string_to_int(labels)
                self.labels_int_to_str = labels_int_to_str

        # Se le label sono array (es. coordinate bidimensionali o più dimensioni)
            if isinstance(labels[0], (list, np.ndarray)):
                #if np.array(labels).ndim == 2:  # Bidimensionale (es. coordinate x, y)
                if np.array(labels).ndim in (1, 2): 
                    self.labels = torch.FloatTensor(np.array(labels))
                elif np.array(labels).ndim > 2:  # Multi-dimensionale (es. mappe 3D)
                    self.labels = torch.FloatTensor(np.array(labels))
                else:
                    raise ValueError(f"Label single-label ha una forma non gestita: {np.array(labels).shape}")
                   # Se le label sono intere (scalari)
            elif isinstance(labels[0], (int, np.integer)):  # Label scalare
                self.labels = torch.LongTensor(labels)
            else:
                raise ValueError(f"Formato delle label non supportato: {type(labels[0])}")
                

            # Nel caso multilabel produco un tensore per ogni tipo di variabile
            # e poi li metto in un dizionario
            # Ottieni i nomi delle label multi-label (es. 'Position', 'Direction')
        else:
            self.label_names = list(self.dataset_original.trials[0].label.keys())

            # Dizionario in cui mettere le label convertite
            labels = {}

            # Dizionario in cui mettere i vari dizionari per la conversione da int a str (se multipli)
            labels_int_to_str ={}
            
            # Ciclo su ogni tipo di label nella lista
            for label_name in self.label_names:

                # Creo una lista con le label relative a questa chiave
                labels_i = [trial.label[label_name] for trial in self.dataset_original.trials]
                
                # Controllo se sono stringhe e nel caso le converto
                if self.labels_format[label_name] == 'string':

                    labels_i, labels_int_to_str_i = convert_labels_string_to_int(labels_i)
                    labels_int_to_str[label_name] = labels_int_to_str_i

                 # Se sono array bidimensionali o multi-dimensionali
                if isinstance(labels_i[0], (list, np.ndarray)):
                    if np.array(labels_i).ndim == 2: ### bidim
                        labels[label_name] = torch.FloatTensor(np.array(labels_i))
                    elif np.array(labels_i).ndim > 2: # multidim
                       labels[label_name] = torch.FloatTensor(np.array(labels_i))
                    else:
                        raise ValueError(f"Label '{label_name}' ha una forma non gestita: {np.array(labels_i).shape}")

                # Se sono intere (scalari)
                elif isinstance(labels_i[0], (int, np.integer)):
                     labels[label_name] = torch.LongTensor(labels_i)  # Interi

                ## Le salvo nel dizionario globale
               # labels[label_name] = labels_i
            
            # Salvo le label nella classe
            self.labels = labels

            # Salvo il dizionario di conversione se non è vuoto
            if len(labels_int_to_str) > 0:
                self.labels_int_to_str = labels_int_to_str   

            #print(f"Labels (Position): Shape: {self.labels['Position'].shape}")
            #print(f"Labels (Direction): Shape: {self.labels['Direction'].shape}")
    def to_device(self, device):

        # Controllo se labels è stato inizializzato
        if not hasattr(self, 'labels'):
            raise AttributeError("L'attributo 'labels' non è stato inizializzato. Verifica che 'create_labels' sia stato chiamato correttamente.")

        # Sposto i dati sul device
        self.eeg_signals = self.eeg_signals.to(device)

        # Sposto le label sul device
        if self.labels_type == 'single_label':
            self.labels = self.labels.to(device)
        else:
            for key in self.labels:
                self.labels[key] = self.labels[key].to(device)
