import numpy as np
import pickle as pkl
import matplotlib.cm as cm
from typing import List
from matplotlib import pyplot as plt
from helpers.eeg_utils import check_number_timepoints, check_labels_type

# Questa è la classe più generale con cui descrivere un singolo trial di dati.
# Contiene l'attività EEG, la label del trial e i tempi. Se necessario, può
# essere ampliata in modo da contenere ulteriori informazioni.
class TrialEEG:

    def __init__(self, eeg_signals, labels, timepoints):
        """
        Args:
            eeg_signals (array-like): Matrice EEG (num_channels x num_timepoints).
            labels (list of tuples): Lista di tuple (nome_etichetta, array_etichetta).
            timepoints (array-like): Timepoints associati ai dati EEG.
        """
        self.eeg_signals = np.array(eeg_signals, dtype=np.float32)
        # Dizionario con nomi e etichette
        self.labels = {name: np.array(label, dtype=np.float32) for name, label in labels} 
        self.num_channels, self.num_timepoints = self.eeg_signals.shape
        self.timepoints = np.array(timepoints, dtype=np.float32)

    def __str__(self):
        """
        Stringa di descrizione dell'oggetto TrialEEG.
        """
        labels_info = "\n".join([f"  {key}: shape {value.shape}" for key, value in self.labels.items()])
        return (f'Numero canali = {self.num_channels}, numero timepoints = {self.num_timepoints}\n'
                f'Istante iniziale = {self.timepoints[0]}, tempo finale = {self.timepoints[-1]}\n'
                f'Labels:\n{labels_info}')

    def plot(self, split_channels=True):

        if split_channels:

            # Mostro i segnali uno sotto l'altro
            
            # La separazione è pari a 4 deviazioni standard in modo che non si sovrappongano
            std = np.std(self.eeg_signals)
            tick_pos = []
            tick_labels = []

            for i in range(self.num_channels):

                y_shift = 4 * std * i
        
                plt.plot(self.timepoints, self.eeg_signals[i,:] + y_shift, color='k', linewidth=0.5)
                
                tick_pos.append(y_shift)
                tick_labels.append(str(i+1))

            plt.yticks(tick_pos, tick_labels)
            plt.ylabel('Channel')

        else:

            # Se invece devo mostrarli tutti assieme, uso un color coding
            # per differenziale i vari canali
            cmap = cm.get_cmap('jet') 
            for i in range(self.num_channels):
                color = cmap(i/(self.num_channels - 1))
                plt.plot(self.timepoints, self.eeg_signals[i,:], color=color, linewidth=0.5, alpha=0.5)

        # Alcune cose non dipendono dal tipo di plot
        plt.xlabel('Time')


class DatasetEEG():

    def __init__(self, trials: List[TrialEEG], info=None):

        self.trials = trials

        # Dizionario con le info sul dataset facoltative
        self.info = info

        # Altre info necessarie
        self.num_trials = len(trials)
        self.num_channels = trials[0].eeg_signals.shape[0]
        self.num_timepoints = check_number_timepoints(trials)
            # Calcola il numero minimo e massimo di timepoints tra i trial
        self.min_timepoints = min(trial.num_timepoints for trial in trials)
        self.max_timepoints = max(trial.num_timepoints for trial in trials)

        # Controlla tipo e formato delle etichette
        self.labels_type, self.labels_format = check_labels_type(trials)
        #print(self.labels_type, self.labels_format)
    
    # Funzione per salvare il dataset su file (per ora con pickle, in futuro con altri formati)
    def save(self, filepath):
        with open(filepath,'wb') as f:
            pkl.dump(self, f)

    # Funzione per caricare (statica perché prima di caricare il dataset non esiste)
    @staticmethod
    def load(filepath):
        with open(filepath,'rb') as f:
            dataset = pkl.load(f)
        return dataset

    # Mostro le info del dataset quando chiamo "print" su di esso
    def __str__(self):
        info_string = f"{'num_trials':25}:  {self.num_trials}\n"
        info_string += f"{'num_channels (per trial)':25}:  {self.num_channels}\n"

        # Rendi sempre esplicito che si tratta di timepoints per trial
        if self.min_timepoints == self.max_timepoints:
            info_string += f"{'timepoints (per trial)':25}:  {self.min_timepoints} (fix)\n"
        else:
            info_string += f"{'timepoints (per trial)':25}:  Min {self.min_timepoints}, Max {self.max_timepoints}\n"

        info_string += f"{'labels_type':25}:  {self.labels_type}\n"
        info_string += f"{'labels_format':25}:  {self.labels_format}\n"
        print(f"Labels type in __str__: {self.labels_type}")
        if self.info:
            for key in self.info:
                info_string += f'{key:<25}:  {self.info[key]}\n'
        return info_string
    #def __str__(self):
        #info_string = f"{'num_trials':25}:  {self.num_trials}\n"
        #info_string += f"{'num_channels':25}:  {self.num_channels}\n"
        #info_string += f"{'num_timepoints':25}:  {self.num_timepoints}\n"
        #info_string += f"{'labels_type':25}:  {sDelf.labels_type}\n"
        #info_string += f"{'labels_format':25}:  {self.labels_format}\n"

        #if self.info:
        #    for key in self.info:
       #         info_string += f'{key:<25}:  {self.info[key]}\n'
       # return info_string
       
   

    # Estrae un subset di trial e crea un nuovo dataset a partire da questi
    def extract_subset(self, idx):

        trials_subset = [self.trials[i] for i in range(self.num_trials) if i in idx]
        return DatasetEEG(trials_subset, self.info)
    
    # Splitta il dataset in training e validation
    def split_dataset(self, validation_size=0.2):

        # Quanti trial prendere
        split_idx = int(np.round((1-validation_size) * self.num_trials))

        # Li prendo a caso e non sequenzialmente
        random_indices = list(np.random.permutation(self.num_trials))
        train_indices = random_indices[0:split_idx]
        validation_indices = random_indices[split_idx:]

        # Creo i due dataset
        dataset_train = self.extract_subset(train_indices)
        dataset_validation = self.extract_subset(validation_indices)

        return dataset_train, dataset_validation
    
    # puoi pensar di metterci un decoratore
    # @staticmethod
    def create_windows(self, window, shift):
       """
       Crea finestre temporali dai dati EEG del dataset.

       Args:
           window (int): Dimensione della finestra temporale (numero di timepoints).
           shift (int): Scostamento tra finestre consecutive.

       Returns:
           DatasetEEG: Nuovo dataset contenente finestre temporali.
       """
       trials_new = []

       for trial in self.trials:
           signals = trial.eeg_signals
           times = trial.timepoints
           labels = trial.labels  # Dizionario delle etichette

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
