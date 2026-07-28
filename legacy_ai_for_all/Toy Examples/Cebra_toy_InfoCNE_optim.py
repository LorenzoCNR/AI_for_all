#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 21:30:49 2024

@author: zlollo
"""

import os
#os.getcwd()
################################## CAMBIARE ########################################
main_folder=r'/home/zlollo/CNR/git_out_cebra/'
os.chdir(main_folder)
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
# Definizione del modello e dell'ottimizzatore
model = nn.Linear(5, 2)
optimizer = optim.SGD(model.parameters(), lr=0.001)
### definisco n modello non lineare

class NonLinearModel(nn.Module):
    def __init__(self):
        super(NonLinearModel, self).__init__()
        self.layer1 = nn.Linear(5, 10)  # Un esempio di aumento della dimensione
        self.activation1 = nn.ReLU()  # Aggiunta di una funzione di attivazione non lineare
        self.layer2 = nn.Linear(10, 5)  # Riduzione della dimensione di nuovo a 5 per mantenere la consistenza
        self.activation2 = nn.ReLU()  # Un'altra funzione di attivazione non lineare
        self.layer3 = nn.Linear(5, 2)  # Riduzione ulteriore per l'output

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.layer3(x)
        return x


# Funzione per inizializzare tutti i pesi e i bias
def init_weights(m):
    if type(m) == nn.Linear:
        # Inizializza i pesi con una distribuzione normale
        init.normal_(m.weight, mean=0.0, std=0.1)
        # Inizializza i bias a zero
        init.constant_(m.bias, 0)

# Applica la funzione di inizializzazione a tutti i moduli del modello


# Creazione del modello
model = NonLinearModel()
print(model)

model.apply(init_weights)


# Definisco 4 tensori come sample di dati (ref=anchor, 1 esempio positivo e 2 neg)
ref = torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.float32)  
pos = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)  
neg1 = torch.tensor([[1, 0, 0, 1, 1]], dtype=torch.float32) 
neg2 = torch.tensor([[1, 1, 0, 0, 1]], dtype=torch.float32)  

### metto insieme gli input
inputs = torch.cat([ref, pos, neg1, neg2], dim=0)  



# Funzione per calcolare la cosine similarity
def psi(x, y):
    x_norm = x / x.norm(dim=1, keepdim=True)
    y_norm = y / y.norm(dim=1, keepdim=True)
    ### traspongo in quanto dovo allineare matrici 2d
    return torch.mm(x_norm, y_norm.transpose(0, 1))

# Funzione di perdita contrastiva 
def expected_contrastive_loss(model, ref, pos, negs):
    # Calcola le rappresentazioni apprese per i campioni
    ref_rep = model(ref)
    pos_rep = model(pos)
    neg_reps = model(negs)
    
    # Calcola la similarità positiva e la somma logaritmica delle esponenziali
    #delle similarità negative
    pos_similarity = torch.exp(psi(ref_rep, pos_rep))
    neg_similarity = torch.exp(psi(ref_rep, neg_reps))
    neg_similarity_sum = torch.logsumexp(neg_similarity, dim=1)

    # Calcola la perdita contrastiva come aspettativa
    loss = -torch.log(pos_similarity.diag() + 1e-9) + neg_similarity_sum  # Aggiunto un piccolo valore per stabilità numerica
    return loss.mean()

# Ciclo di addestramento
for epoch in range(100):
    optimizer.zero_grad()
    loss = expected_contrastive_loss(model, ref, pos, inputs[2:])
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.grad.norm())
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
