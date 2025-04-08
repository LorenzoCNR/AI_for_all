import torch
import torch.nn as nn
import torch.optim as optim
from models.base_models import BaseModel
import torch.nn.functional as F
#from torchviz import make_dot

''' Logic (Computationa graph)
     Input x  
        │  
     Encoder (self.layers)  
        │  
     Embeddings f_x (requires_grad=True)  
        │  
     Cosine Similarity (ψ)  
        │  
     Weighted Similarity (ψ + log(1 + w))  
        │  
     ┌────────────┬──────────────┐  
     │            │  
 Alignment Loss  Uniformity Loss  
     │            │  
  SUM (Final Loss)  
        │  
 Backpropagation (loss.backward())  
        │  
 Updated Encoder Parameters  


'''
'''
###### scriv una loss a secodn del problema che vuoi affrontare...
ref: 
    -representation learnign with contrastive predictive coding (time repres)
   - SOFT CONTRASTIVE LEARNING FOR TIME SERIES
   - Supervised Contrastive Learning
   - https://medium.com/@juanc.olamendy/contrastive-learning-a-comprehensive-guide-69bf23ca6b77 (rassegna sul CL)
   -  Multi-Label Contrastive Learning : A Comprehensive Study
   - ss cookbook
   -  StatioCL: Contrastive Learning for Time Series via Non-Stationary and Temporal Con
   -  bishop 
   - murphy
   -  DYNAMIC CONTRASTIVE LEARNING for time series repr
    
'''

# Contrastive encoder with weighted batches
# the encoder uses batch of data straight with no negative or positive samples
class EncoderContrastiveWeights(BaseModel):
    def __init__(self, layers: nn.Module, labels_distance=None, labels_weights=None, 
                 temperature=1, train_temperature=False, loss_type="weighted_contrastive",
                 positive_offset=1, positive_window=0):        
        super().__init__()
        #super(EncoderContrastiveWeights, self).__init__()
        ## Network layers
        self.layers = layers
        # Function for computing label distances
        self.labels_distance = labels_distance
        self.train_temperature = train_temperature
        self.labels_weights = nn.Parameter(torch.tensor(labels_weights), requires_grad=False)
        #else:
         #   self.labels_weights = None
        self.positive_offset = positive_offset
        self.positive_window = positive_window
        ### batch counter (for debug)
        #self.batch_counter = 0
        
        ## choose loss
        self.loss_type = loss_type
        
        if train_temperature:
            self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.temperature_min = torch.tensor(0.0001)
        else:
            self.temperature = torch.tensor(temperature)

    def forward(self, x):
        f_x = self.layers(x)
        if self.training:  # Solo in training!
        
            f_x.retain_grad()  
       # Normalizzo ( opzionale, lo faccio nel modello)
        #f_x = torch.nn.functional.normalize(f_x, dim=1, p=2)
        return f_x

   
    # La loss in questo caso riceve solo f_x, poiché deve poi prendere ogni sample
    # del batch come riferimento e tutti gli altri come esempi positivi/negativi
    # different from standard infonce which is the  negative of the log of
    # a ratio between positive pairs and all pairs
    def loss(self, f_x, weights):
       return self.compute_loss(f_x, weights)
   
    def compute_loss(self, f_x, weights, labels=None):
        """ """
        if self.loss_type == "infoNCE":
            return self.infoNCE_loss(f_x)
        elif self.loss_type == "weighted_contrastive":
            if self.labels_distance is None or self.labels_weights is None:
                raise ValueError("Weigthed loss needs labels and weights.")
            weights = self.labels_distance.get_weights(labels)
            return self.weighted_contrastive_loss(f_x, weights)
        elif self.loss_type== "time_contrastive":
            return self.time_contrastive_loss(f_x)
        elif self.loss_type == "supervised_contrastive":
            return self.supervised_contrastive_loss(f_x, weights)
        else:
            raise ValueError(f" unsupported Loss type {self.loss_type} ")
     
        
    #def infoNCE_loss(f_x):
    def time_contrastive_loss(self, f_x):
        # if temperature has to be trained
        batch_size = f_x.shape[0]
        temperature = torch.exp(self.temperature) if self.train_temperature else self.temperature
        sim_matrix = torch.einsum('ai,bi->ab', f_x, f_x) / temperature
        '''
        return loss.mean()
          prendo gli embedding, sceglo i positivi (uno o più intorno ad un campione ancora)
          che qui non viene scelto casulamente ma si passa per tutta la sequenza 
          con attenione a non sforare ...si fa quindi einsum per i positivi e li si 
          fraziona con i negativi che sono gli altri...qualche mask
          
          '''
        mask = torch.eye(batch_size, dtype=torch.bool, device=f_x.device)  
        if self.positive_window > 0:
            self.positive_offset=None
            for i in range(1, self.positive_window + 1):
                ## creo una matrice identita' di dimensione batch per batch
                ## valori booleani (per escludere elementi sulla diagonale...similarita' con se' stesso)
                ##  mask sullo stesso device
                mask |= torch.eye(batch_size, batch_size, dtype=torch.bool, device=f_x.device).roll(shifts=i, dims=1)
                mask |= torch.eye(batch_size, batch_size, dtype=torch.bool, device=f_x.device).roll(shifts=-i, dims=1)
        elif self.positive_offset:
        # Se uso offset fisso
               #print(f" Test: f_x.requires_grad = {f_x.requires_grad}")
                mask |= torch.eye(batch_size, dtype=torch.bool, device=f_x.device).roll(shifts=self.positive_offset, dims=1)
                mask |= torch.eye(batch_size, dtype=torch.bool, device=f_x.device).roll(shifts=-self.positive_offset, dims=1)


        # positives example defined in mask
        positives = sim_matrix[mask].view(batch_size, -1).sum(dim=1)  
        #print(f" Max absolute difference after requires_grad_: {diff.item()}")

        print(f" positive examples are: {positives}")
        # Escludiamo i positivi dal denominatore
        negatives = torch.exp(sim_matrix) * (~mask)
        negatives_sum = negatives.sum(dim=1)
        print(f" negative sum is : {negatives_sum}")
        print(mask.int())  # Converti in interi per vedere chiaramente
       # print(negatives_sum)
        # Calcoliamo la loss
        eps = 1e-10
        loss = -torch.log(positives+eps/ negatives_sum+eps)
        print(f_x.norm(dim=1))  # Dovrebbe essere ~1 se ben normalizzato
        
  
          ##https://einsum.joelburget.com/
         
        negatives_sum = negatives.sum(dim=1) + 1e-10
        return loss.mean()     
     
    def weighted_contrastive_loss(self,f_x, weights):
        
            '''
            compute contrstive loss based on embedding and distance weights
    
    cfr papers:
   - Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere
   - Alignment-Uniformity aware Representation Learning for Zero-shot Video Classification
             '''
             # if temperature has to be trained
            temperature = torch.exp(self.temperature) if self.train_temperature else self.temperature
    
            # build a tensor with similarities (cosine dot similarity)
            # N.B. Similarities between embeddings!
            psi = torch.einsum('ai,bi->ab', f_x, f_x) / temperature
            
            #print("\n Matrice di similarità tra embedding (psi):\n", psi)
           
           # Prevent numerical issues with extremely negative values
           # psi[psi < -1e10] = 0
    
            # Mask diagonal elements to ignore self-similarity
            # Stabilize diagonal elements
            psi.fill_diagonal_(-1e15)
    
            # add 1 to avoid 0
            # could also add a numebr close to zero (i.e. 1e-6)
            # Apply log transformation to avoid numerical issues
            weights_log = torch.log(1+ weights)
            #print(f"psi size is: {psi.shape}")
            #print(f"weights_log size is: {weights_log.shape}")
           
            # Weighted similarity matrix
            psi_weighted = psi.view((*psi.shape, 1)) + weights_log
            print("\n Matrice pesata (psi_weighted):\n", psi_weighted)
            # Applico logsumexp ad entrambi e calcolo separatamente i due termini della loss
            #alignement = -torch.sum(torch.logsumexp(psi_weighted, dim=1))
            #uniformity = labels_num * torch.sum(torch.logsumexp(psi, dim=1))
            # Compute alignment and uniformity terms
            total_label_weight = self.labels_weights.sum()
               
            alignement = -torch.sum(self.labels_weights * torch.sum(torch.logsumexp(psi_weighted, dim=1), dim=0))
            uniformity = total_label_weight * torch.sum(torch.logsumexp(psi, dim=1))
            
            # psi_weighted_inv = psi.view((*psi.shape, 1)) + torch.log(1 - weights + 1e-10)
            # den = torch.sum(torch.logsumexp(psi_weighted_inv, dim=1))
    
            # total loss
            loss = alignement + uniformity
            
            return alignement, uniformity, loss
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        """
        forward pass, loss computation weight upadate
                
        """
        x, labels = batch
        print(f"Chiavi in labels_distance_functions: {self.labels_distance.labels_distance_functions.keys()}")
        print(f"Chiavi in labels: {labels.keys()}")
         # Filter labels to only keep those used in distance computations
        # **Select labels only if loss requests*
        if self.loss_type in ["weighted_contrastive", "supervised_contrastive"] and self.labels_distance is not None:
            print(f"Labels before filtering: {labels}")
            
            labels = {k: v for k, v in labels.items() if k in self.labels_distance.labels_distance_functions}
            weights = self.labels_distance.get_weights(labels)
            if weights is None:
                raise ValueError(" Errore: weights è None! Controlla le etichette nel batch.")
        else:
           weights = None
        #print(f"\n Labels dopo il filtraggio: {labels.keys()}")  
        # Reste gradients (if not in evaluation mode)
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass       
        f_x = self.forward(x)
        
        #weights = self.labels_distance.get_weights(labels)

        loss = self.loss(f_x, weights)

        
     
        # Perform backpropagation only if in training mode
        if not is_eval: 
            #loss.backward()
            ### Retain Graoh for debugging
            loss.backward(retain_graph=True)  

            optimizer.step() 
    
            
        return {'loss': loss.item()}

    
