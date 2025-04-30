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
    def __init__(self, layers: nn.Module, labels_distance, labels_weights, temperature=1,
                 train_temperature=False, loss_type="weighted_contrastive"):
        
        super().__init__()
        #super(EncoderContrastiveWeights, self).__init__()
        ## Network layers
        self.layers = layers
        # Function fto compute label distances
        self.labels_distance = labels_distance
        # True/False scale parameter
        self.train_temperature = train_temperature
        #
        #self.use_temporal = use_temporal
        #
        #self.use_labels = use_labels
        #
        #self.temporal_window = temporal_window  
        
        
        '''
        loss_type: "infonce". "weighted_contrastive", "supervised_contrastive", 
        "alignment_uniformity"
        '''
        self.loss_type= loss_type
        self.labels_weights = nn.Parameter(torch.tensor(labels_weights), requires_grad=False)
        ### batch counter (for debug)
        #self.batch_counter = 0
        
        if train_temperature:
            self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            #self.temperature_min = torch.tensor(0.0001)
        else:
            self.temperature = torch.tensor(temperature)

    def forward(self, x):
        f_x = self.layers(x)
        if self.training:  # Solo in training!
            f_x.retain_grad()  
        return f_x

        # Save original values before enabling requires_grad
        #f_x_before = f_x.clone().detach()  # Clone to prevent gradient tracking
        # Ensure gradients are tracked for f_x
        #print(f" Test: f_x.requires_grad = {f_x.requires_grad}")
         #f_x.requires_grad_(True)
       # print(f" Test: f_x.grad_fn = {f_x.grad_fn}")
        # Normalizzo
        #print(f"f_x.requires_grad = {f_x.requires_grad}")
        #print(f"f_x.grad_fn = {f_x.grad_fn}")
        #f_x = F.normalize(f_x, dim=1, p=2)
        # Save values after setting requires_grad
        #f_x_after = f_x.clone().detach()
        # Check if anything changed
        #diff = (f_x_before - f_x_after).abs().max()
        #print(f" Max absolute difference after requires_grad_: {diff.item()}")
            
        #f_x = torch.nn.functional.normalize(f_x, dim=1, p=2)
        #check  PyTorch calcoli il gradiente
        # Retain gradient for debugging
        #f_x.retain_grad()

        return f_x
    
    def loss(self, f_x, weights):
     """
    
     """
     return self.compute_loss(f_x, weights)
    # La loss in questo caso riceve solo f_x, poiché deve poi prendere ogni sample
    # del batch come riferimento e tutti gli altri come esempi positivi/negativi
    # different from standard infonce which is the  negative of the log of
    # a ratio between positive pairs and all pairs
    def compute_loss(self, f_x,labels):
        
      """
       Compute the chosen contrastive loss
      """
      weights = self.labels_distance.get_weights(labels)
      
      if self.loss_type == "infoNCE":
            return self.infoNCE_loss(f_x)
      elif self.loss_type == "weighted_contrastive":
            return self.weighted_contrastive_loss(f_x, weights)
      elif self.loss_type == "supervised_contrastive":
            return self.supervised_contrastive_loss(f_x, weights)
      else:
            raise ValueError(f"Loss type {self.loss_type} unsupported")

    def infoNCE_loss(self, f_x, time_indices, offset=10):
        batch_size = f_x.shape[0]
        temperature = torch.exp(self.temperature) if self.train_temperature else self.temperature
    
        # Compute cosine similarities
        sim_matrix = torch.einsum('ai,bi->ab', f_x, f_x) / temperature
    
        # Define positive samples based on temporal offset
        offset = 10 
        positives = torch.diag(sim_matrix, offset=offset)  # Positivi a distanza di "offset"
        
        # exclude extreme points  (which cold not have offse)
        mask = torch.arange(batch_size - offset)  
        positives = positives[mask]
    
        # Compute loss
        negatives = sim_matrix  # Tutti gli altri sono negativi
        loss = -torch.log(positives / (torch.sum(torch.exp(negatives), dim=1) - torch.exp(positives)))
    
        return loss.mean()
    


    def weighted_contrastive_loss(self, f_x, weights):
        """
        Contrastive loss based on label distance.
        """
        # VARI PRINT PER DEBUG
        #print(f"Labels Weights Shape: {self.labels_weights.shape}")
        #print(f"Labels Weights Values: {self.labels_weights}")
        #print(f"Shape di weights (distanze delle etichette): {weights.shape}")  # Deve essere [batch_size, batch_size, 2]
        #labels_num = weights.shape[2]
        
       
        ### trainable temperature
        temperature = torch.exp(self.temperature) if self.train_temperature else self.temperature
        # build a tensor with similarities (cosine dot similarity)
        # N.B. Similarities between embeddings!       
        psi = torch.einsum('ai,bi->ab', f_x, f_x) / temperature
        # Prevent numerical issues with extremely negative values
        print(psi.shape)
        # psi[psi < -1e10] = 0
        print("\n Matrice di similarità tra embedding (psi):\n", psi)
        print(f"psi size is: {psi.shape}")

        # Mask diagonal elements to ignore self-similarity
        # Stabilize diagonal elements
        #psi.fill_diagonal_(-1e15)
        
        ### add 1 or a small epsilon to avoid log of zero
        weights_log = torch.log(1 + weights)
        print(f"weights_log size is: {weights_log.shape}")
        #print(f"weights size is: {weights.shape}")
        #print(f"layers size is: {f_x.shape}")

        #psi_weighted = psi.unsqueeze(2) + weights_log 
        
        ### per tensori contigui usa questa
        psi_weighted = psi.view((*psi.shape, 1)) + weights_log

        total_label_weight = self.labels_weights.sum()
        #print("\n Matrice combinata delle distanze tra le etichette:\n", weights)
        #print("\n Matrice pesata (psi_weighted):\n", psi_weighted)
        
        alignement = -torch.sum(self.labels_weights * torch.sum(torch.logsumexp(psi_weighted, dim=1), dim=0))
        
        #
        uniformity = total_label_weight * torch.sum(torch.logsumexp(psi, dim=1))
        #uniformity = torch.sum(torch.logsumexp(psi, dim=1))
        loss = alignement + uniformity

        return alignement, uniformity, loss
    
    # def supervised_contrastive_loss(self, f_x, weights):
    #     """
    #     Supervised Contrastive Loss basata su etichette.
    #     """
    #     temperature = torch.exp(self.temperature) if self.temperature.requires_grad else self.temperature

    #     # Similarità tra gli embedding
    #     sim_matrix = torch.einsum('ai,bi->ab', f_x, f_x) / temperature

    #     # Applica i pesi solo ai positivi
    #     positive_mask = (weights > 0).float()
    #     positive_sim = positive_mask * sim_matrix

    #     # Loss supervisionata
    #     loss = -torch.log(torch.sum(torch.exp(positive_sim), dim=1) / torch.sum(torch.exp(sim_matrix), dim=1))
    #     return loss.mean()

    
    
    def process_batch(self, batch, optimizer, is_eval=False):
        
        """
        forward pass, loss computation weight upadate
        
        
        """
        
        x, labels = batch
        
        # Filter labels to only keep those used in distance computations
        labels = {k: v for k, v in labels.items() if k in self.labels_distance.labels_distance_functions}

        print(f"\n Labels dopo il filtraggio: {labels.keys()}")  
        
        # Reste gradients (if not in evaluation mode)
        if not is_eval: 
            optimizer.zero_grad()

        # Forward pass       
        f_x = self.forward(x)
        
    # TEST 1: Verifica se `f_x` partecipa al grafo computazionale
        # print(f" Test 1: f_x.requires_grad = {f_x.requires_grad}")
        #print(f" Test 1: f_x.grad_fn = {f_x.grad_fn}")


         # Compute label weights
        weights = self.labels_distance.get_weights(labels)

        # Loss
        alignement, uniformity, loss = self.loss(f_x, weights)
        # Debugging: Check computational graph
        #dot = make_dot(loss, params=dict(self.layers.named_parameters()))
        #dot.render("computation_graph_v2", format="pdf")  # Salva il grafo in un file PNG
        #dot.view() 
        
       
        


        # Perform backpropagation only if in training mode
        if not is_eval: 
            #loss.backward()
            ### Retain Graoh for debugging
            loss.backward(retain_graph=True)  
            #print(f"Gradient on f_x: {f_x.grad if f_x.grad is not None else 'None'}")
            
            # for name, param in self.layers.named_parameters():
            #     if param.grad is None:
            #         print(f"{name}: grad = {param.grad.norm() if param.grad is not None else 'None'}")
            #         print(f" Gradiente non aggiornato per: {name}")
            #         print(f"{name}: grad norm = {param.grad.norm().item()}")
          
            optimizer.step() 
            # Check if embeddings are receiving gradients
        #     if f_x.grad is None:
        #         print("⚠️ Warning: `f_x.grad` is None! Checking computation graph...")
        #         # Validate if loss is connected to the computational graph
        #         print(f"loss.grad_fn: {loss.grad_fn}")
        #         for name, param in self.layers.named_parameters():
        #             print(f"Test: Param {name}, grad norm: {param.grad.norm() if param.grad is not None else 'None'}")
        # print(f"Epoch  Loss = {loss.item()}")
            
        return {'alignement': alignement.item(),
                'uniformity': uniformity.item(),
                'loss': loss.item()}

    
