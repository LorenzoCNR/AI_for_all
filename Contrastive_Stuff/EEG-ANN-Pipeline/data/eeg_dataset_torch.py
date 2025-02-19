import torch
import numpy as np
from data import DatasetEEG
from torch.utils.data import Dataset
from helpers.eeg_utils import convert_labels_string_to_int



class DatasetEEGTorch(Dataset):

    def __init__(self, dataset: DatasetEEG, selected_labels=None):
        
        # Controllo che i trials siano tutti della stessa lunghezza
        # altrimenti non posso creare il dataset pytorch
        if not isinstance(dataset.num_timepoints, int):
            raise ValueError('I trial non hanno tutti la stessa lunghezza')
        
        # Mantengo un riferimento al dataset da cui previene
        # e altre informazioni utili
        self.dataset_original = dataset
        self.labels_type = dataset.labels_type
        self.labels_format = dataset.labels_format
        print( self.labels_format)
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
        self.create_labels(selected_labels)

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
    
    def create_labels(self, selected_labels=None):

        # Se lavoro con tutte le labels del dataset di partenza, posso salvare le info
        if selected_labels is None:
            self.labels_type = self.dataset_original.labels_type
            self.labels_format = self.dataset_original.labels_format

            if self.labels_type == 'multi_label':
               self.label_names = [label_name for label_name in self.labels_format]
        else:
    # Altrimenti seleziono solo quelle che mi interessano
            if type(selected_labels) is not list: selected_labels = [selected_labels]
     
            self.labels_format = {label_name: self.dataset_original.labels_format[label_name] for label_name in selected_labels}
            self.label_names = [label_name for label_name in self.labels_format]

    # Se dopo la restrizione è cambiato in single-label, devo modificare le info
        if len(self.labels_format) == 1:
            self.labels_type = 'single_label'
            self.label_names = self.label_names[0]
            self.labels_format = self.labels_format[self.label_names]
        else:
            self.labels_type = 'multi_label'

        # Separo il caso single-label da quello multi-label

        # Caso single label
        if self.labels_type == 'single_label':

            # Creo una lista con le label
            # Se partivo da un dataset multilabel devo usare i dizionari
            if selected_labels is not None:
                labels = [trial.labels[self.label_names] for trial in self.dataset_original.trials]
            else:
                # Altrimenti posso prendere direttamente la label
                labels = [trial.labels for trial in self.dataset_original.trials]

            #labels = [trial.labels for trial in self.dataset_original.trials]
            #labels = [list(trial.labels.values())[0] for trial in self.dataset_original.trials]
            #print(labels)
           # Controllo se sono stringhe
            if self.labels_format == 'string':
                labels, labels_int_to_str = convert_labels_string_to_int(labels)
                self.labels_int_to_str = labels_int_to_str
    
            # Converto la lista in un numpy array per ottimizzare la performance
            labels_array = np.array(labels)
    
            # Trasformo le label in un tensore PyTorch
            #if labels_array.dtype == np.float32 or labels_array.dtype == np.float64:
            self.labels = torch.FloatTensor(labels_array)
           # else:
           #self.labels = torch.LongTensor(labels_array)

     #Nel caso multilabel produco un tensore per ogni tipo di variabile
            # e poi li metto in un dizionario      
       
        else:
           

            # Dizionario in cui mettere le label convertite
            labels = {}

            # Dizionario in cui mettere i vari dizionari per la conversione da int a str (se multipli)
            labels_int_to_str = {}
            
            # Ciclo su ogni tipo di label nella lista
            for label_name in self.label_names:

                # Creo una lista con le label relative a questa chiave
                # list comprehension
                labels_i = [trial.labels[label_name] for trial in self.dataset_original.trials]
                labels_array = np.array(labels_i)
                #print(label_name, labels_i)
                # Controllo se sono stringhe e nel caso le converto
                if self.labels_format[label_name] == 'string':

                    labels_array, labels_int_to_str_i = convert_labels_string_to_int(labels_i)
                    labels_int_to_str[label_name] = labels_int_to_str_i

                # Converto in base anche al tipo di dato
                #if self.labels_format[label_name] == 'float':
                labels_tensor = torch.FloatTensor(labels_array)
                #else:
                #labels_tensor= torch.LongTensor(labels_array)

                # Le salvo nel dizionario globale
                labels[label_name] = labels_tensor
            
            # Salvo le label nella classe
            self.labels = labels

            # Salvo il dizionario di conversione se non è vuoto
            if len(labels_int_to_str) > 0:
                self.labels_int_to_str = labels_int_to_str            

    def to_device(self, device):

        # Sposto i dati sul device
        self.eeg_signals = self.eeg_signals.to(device)

        # Sposto le label sul device, tenendo conto dei possibili casi
        if self.labels_type == 'single_label':
            self.labels = self.labels.to(device)
        else:
            for key in self.labels:
                self.labels[key] = self.labels[key].to(device)
