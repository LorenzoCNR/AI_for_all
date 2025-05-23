
import torch
import numpy as np
from data import DatasetEEGTorch

#----------------------------- Standard InfoNCE ----------------------------- #

# Permette di samplare indici per costruire i sample positivi
# a partire da una label discreta
class SamplerDiscrete():

    def __init__(self, labels):

        # Se non è un array di numpy, lo converto
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        self.labels = labels

        # Ordino
        self.sorted_idx = torch.from_numpy(np.argsort(self.labels))

        # Distribuzione cumulativa
        self.counts = np.bincount(self.labels)
        self.cdf = np.zeros((len(self.counts) + 1,))
        self.cdf[1:] = np.cumsum(self.counts)

    def sample(self, reference_labels):

        # Converto le label se necessario
        if torch.is_tensor(reference_labels):
            reference_labels = reference_labels.cpu().numpy()

        idx = np.random.uniform(0, 1, len(reference_labels))
        idx *= self.cdf[reference_labels + 1] - self.cdf[reference_labels]
        idx += self.cdf[reference_labels]
        idx = idx.astype(int)

        return self.sorted_idx[idx]
    

# Permette di samplare indici per costruire i sample positivi
# a partire da una label continua
class SamplerContinuousGaussian():

    def __init__(self, labels, std=1):

        # Se non è un array di numpy, lo converto
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()

        # Se la dimensione è uno, li espando
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=-1)

        self.labels = labels
        self.std = std

        # Precalcolo i quadrati per semplificare il conto dopo
        self.labels_squared = self.labels**2

    def sample(self, reference_labels):

        # Converto le label se necessario
        if torch.is_tensor(reference_labels):
            reference_labels = reference_labels.cpu().numpy()

        # Se la dimensione è uno, li espando
        if reference_labels.ndim == 1:
            reference_labels = np.expand_dims(reference_labels, axis=-1)   

        # Samplo da una gaussiana
        samples = np.random.randn(*reference_labels.shape) * self.std + reference_labels

        # So che "self.labels" ha dimensione (N_trials, 1)
        # e che "reference_values" ha dimensione (batch_size, 1)
        # Voglio la distanza tra ogni elemento di reference_values e ogni 
        # elemento di self.labels. Quindi sfruttando un po' di trucchetti
        # di broacasting posso calcolare (x - y)^2 = x^2 + y^2 - 2xy
        samples_squared = (samples**2).T

        product_term = np.einsum('ni,mi->nm', self.labels, reference_labels)
        distance_matrix = self.labels_squared + samples_squared - 2 * product_term

        # Data la matrice di distanze, cerco gli elementi che la minimizzano per ogni reference
        return np.argmin(distance_matrix, axis=0)


# Il contrastive batch sampler deve fornire tre tensori, x, y_pos e y_neg
# tutti della stessa dimensione batch_size x channels x timepoints
# Per farlo genera prima gli esempi negativi e le reference casualmente
# Poi deve avere un criterio con cui samplare i positivi
class DataLoaderContrastive():

    def __init__(self, dataset: DatasetEEGTorch, samplers, batch_size, batches_per_epoch=1):

        self.dataset = dataset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.samplers = samplers

    def sample_positive(self, reference_idx):

        if self.dataset.labels_type == 'single_label':     
            reference_labels = self.dataset.labels[reference_idx]
            return self.samplers.sample(reference_labels)
        else:
            positive_samples = dict()
            for label_name in self.samplers:
                reference_labels = self.dataset.labels[label_name][reference_idx]
                positive_samples[label_name] = self.samplers[label_name].sample(reference_labels) 
            return positive_samples
        
    def __iter__(self):

        for _ in range(self.batches_per_epoch):

            # Genero i numeri casuali per reference e negative
            rand_idx = np.random.choice(self.dataset.num_trials, self.batch_size * 2)

            # Prendo gli indici di reference e negative
            reference_idx = rand_idx[0:self.batch_size]
            negative_idx = rand_idx[self.batch_size:]

            # Prendo i sample relativi
            x = self.dataset.eeg_signals[reference_idx,:,:,:]
            y_neg = self.dataset.eeg_signals[negative_idx,:,:,:]

            # Per le positive devo samplare adeguatamente
            # Se i positive_idx sono una lista (più label)
            # devo creare un tensore più grande per gli y_pos 
            positive_idx = self.sample_positive(reference_idx)

            if self.dataset.labels_type == 'single_label':     
                y_pos = self.dataset.eeg_signals[positive_idx,:,:,:]
                labels = self.dataset.labels[reference_idx]
            else:
                y_pos = {label_name: self.dataset.eeg_signals[positive_idx[label_name],:,:,:] for label_name in positive_idx}
                labels = {label_name: self.dataset.labels[label_name][reference_idx] for label_name in positive_idx}
            yield x, y_pos, y_neg, labels


    def __len__(self):
        return self.batches_per_epoch
    

# ---------------------------- My modified InfoNCE --------------------------- #
import torch


class LabelsDistance:
    def __init__(self, labels_distance_functions):
        """
        Inizializza LabelsDistance.

        Args:
            labels_distance_functions (dict or single function):
                - Dcictionar can be sngle or multi-label (position+direction).
                - Single functon: single label
        """
        #### caso dict (multi o single label)
        if isinstance(labels_distance_functions, dict):
            self.labels_distance_functions = labels_distance_functions
            self.label_keys = list(labels_distance_functions.keys())  
            self.multi_label = len(self.label_keys) > 1 
        else:
            ##  pass the function with no dictionary
            self.labels_distance_functions = labels_distance_functions  #
            self.label_keys = None
            self.multi_label = False  

    def get_weights(self, labels):
        if not labels:
            raise ValueError("Le etichette non sono state fornite o sono vuote.")
            print(f"Received labels: {labels}")
            # Elenco delle etichette per debug
        for label_name, label_values in labels.items():
            if label_values is None:
                print(f"Errore: {label_name} è None!")
    # Continuare con il calcolo dei pesi
        """
       COmpute the matrix distance between labels in batch (weights for loss)
        
        Args:
            labels (dict o tensore singolo): 
                - if multi-label,it is a dict (ex. {"position": tensor(512,2), "direction": tensor(512,)}).
                - if mono-label, can be asingle tensor or a dict with just one key.

        Returns:
            torch.Tensor: distance matrix (batch_size, batch_size, num_labels)
            num labels = 1,2,...N
        """
           
        # def pairwise_tensor_function(f, x):
        #     print(f"pairwise_tensor_function: Input x shape {x.shape}")  # Debug
        #     return f(x, x.T) if x.ndim == 1 else f(x, x)

        def pairwise_tensor_function(f, x):
     
          output = f(x, x.T) if x.ndim == 1 else f(x, x)
        
          return output
    
            #  Se labels è un dizionario (MULTI-LABEL o MONO-LABEL con una sola chiave)
        if isinstance(labels, dict):
            
            
            if len(labels) == 1:  # dictionary with one label
                label_name = next(iter(labels))
                ### get the function from the label
             
                f = (self.labels_distance_functions[label_name]  #
                    if isinstance(self.labels_distance_functions, dict)
                    else self.labels_distance_functions  
                )
                weight = pairwise_tensor_function(f, labels[label_name])
                print(f"\n Matrice di distanza per '{label_name}':\n", weight)
            
                return weight.view((weight.shape[0], weight.shape[1], 1))

            # Multi-label
            weights = []
            for label_name, label_values in labels.items():
                f = self.labels_distance_functions[label_name]  # Prende la funzione giusta dal dizionario
                weight = pairwise_tensor_function(f, label_values)
                print(f"\nMatrice di distanza per '{label_name}':\n", weight)
                weights.append(weight)
            
            return torch.stack(weights, dim=-1)  # (batch_size, batch_size, num_labels)

        # MONO-LABEL with single (ad es. LabelsDistance(adaptive_gaussian_distance))
        # either it uses straight the function or get the dict key
        else:
            f = (
                self.labels_distance_functions
                if callable(self.labels_distance_functions)
                else next(iter(self.labels_distance_functions.values()))  
            )
            
            weight = pairwise_tensor_function(f, labels)
         

            return weight.view((weight.shape[0], weight.shape[1], 1))
