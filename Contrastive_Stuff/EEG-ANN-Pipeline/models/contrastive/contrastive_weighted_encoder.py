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


# Contrastive encoder with weighted batches
# the encoder uses batch of data straight with no negative or positive samples
class EncoderContrastiveWeights(BaseModel):
    def __init__(self, layers: nn.Module, labels_distance, labels_weights, temperature=1, train_temperature=False):
        
        super().__init__()
        #super(EncoderContrastiveWeights, self).__init__()
        ## Network layers
        self.layers = layers
        # Function for computing label distances
        self.labels_distance = labels_distance
        self.train_temperature = train_temperature
        self.labels_weights = nn.Parameter(torch.tensor(labels_weights), requires_grad=False)
        ### batch counter (for debug)
        #self.batch_counter = 0
        
        if train_temperature:
            self.temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
            self.temperature_min = torch.tensor(0.0001)
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

    # La loss in questo caso riceve solo f_x, poiché deve poi prendere ogni sample
    # del batch come riferimento e tutti gli altri come esempi positivi/negativi
    # different from standard infonce which is the  negative of the log of
    # a ratio between positive pairs and all pairs
    def loss(self, f_x, weights):
        
        '''
        compute contrstive loss based on embedding and distance weights

        '''
        #print(f"Labels Weights Shape: {self.labels_weights.shape}")
        #print(f"Labels Weights Values: {self.labels_weights}")
        #print(f"Shape di weights (distanze delle etichette): {weights.shape}")  # Deve essere [batch_size, batch_size, 2]
        #labels_num = weights.shape[2]

        # if temperature has to be trained
        temperature = torch.exp(self.temperature) if self.train_temperature else self.temperature

        # build a tensor with similarities (cosine dot similarity)
        # N.B. Similarities between embeddings!
        psi = torch.einsum('ai,bi->ab', f_x, f_x) / temperature
        
       # print("\n Matrice di similarità tra embedding (psi):\n", psi)
       
       # Prevent numerical issues with extremely negative values
       # psi[psi < -1e10] = 0

        # Mask diagonal elements to ignore self-similarity
        # Stabilize diagonal elements
        psi.fill_diagonal_(-1e15)

        # add 1 to avoid 0
        # could also add a numebr close to zero
        # Apply log transformation to avoid numerical issues
        weights_log = torch.log(1+ weights)
        #print(f"psi size is: {psi.shape}")
        #print(f"weights_log size is: {weights_log.shape}")
        #print(f"weights size is: {weights.shape}")
        #print(f"layers size is: {f_x.shape}")


        # Weighted similarity matrix
        psi_weighted = psi.view((*psi.shape, 1)) + weights_log
        #print("\n Matrice combinata delle distanze tra le etichette:\n", weights)
       # print("\n Matrice pesata (psi_weighted):\n", psi_weighted)
        # Applico logsumexp ad entrambi e calcolo separatamente i due termini della loss
        # alignement = -torch.sum(torch.logsumexp(psi_weighted, dim=1))
        # uniformity = labels_num * torch.sum(torch.logsumexp(psi, dim=1))
        # Compute alignment and uniformity terms
        total_label_weight = self.labels_weights.sum()
        #print(f"Labels Weights Shape: {self.labels_weights.shape}")
       # print(f"Labels Weights Values: {self.labels_weights}")
        ##print(f"Shape di weights (distanze delle etichette): {weights.shape}")  # Deve essere [batch_size, batch_size, 2]
        
        
        # Qui avviene il broadcasting!
        #weighted_weights = self.labels_weights * weights  
        #print(f"Weighted Weights Shape: {weighted_weights.shape}")
        ### doubel summation over batches i different from j of the log sum 
        # loss part link to embedding alignment
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
        
        # Filter labels to only keep those used in distance computations
        labels = {k: v for k, v in labels.items() if k in self.labels_distance.labels_distance_functions}

        #print(f"\n✅ Labels dopo il filtraggio: {labels.keys()}")  
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

    