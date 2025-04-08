# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 14:02:14 2025

@author: loren
"""

import torch
import os
import numpy as np
###  DEFINE DISTANCES to compute the loss
"""
This script contains implementations of various distance functions commonly
 used in data analysis and machine learning:

1. **Adaptive Gaussian Distance**:
   - Computes a normalized distance between two vectors using the (average)
   sample variance.
   - Applies a Gaussian (RBF) kernel to the Euclidean distance, transforming
   it into a similarity measure ranging from 0 to 1.
   - The bandwidth parameter (sigma) controls how quickly the similarity
   decreases with distance.

2. **Minkowski Distance**:
   - Generalization of Euclidean and Manhattan distances.
   - Parameterized by an order 'p', where:
     - p = 1: Equivalent to Manhattan (L1) distance.
     - p = 2: Equivalent to Euclidean (L2) distance.
   - Measures the distance between two points in a normed vector space.

3. **Normalized Minkowski Distance**:
   - Computes the Minkowski distance after normalizing the input vectors.
   - Normalization involves subtracting the mean and dividing by the standard
   deviation for each dimension.
   - Ensures that each feature contributes equally to the distance computation,
   especially when features have different scales or units.

4. **Direction Distance**:
   - Computes a qualitative measure of similarity based on categorical labels.
   - Returns 1 if the labels are identical, otherwise 0.
   - Useful for comparing directional or categorical data where exact
   matches are significant.

5. **Circular Distance**:
   - Computes the distance between angles or directions in a periodic space
   (e.g., compass directions).
   - Accounts for the circular nature of such data, ensuring that the distance
   between angles like 0° and 360° is correctly evaluated as zero.
   - Particularly useful for data representing cyclical patterns, such as 
   time-of-day or compass bearings.
"""


#  ecuclidean gaussian distance (verifica le funzioni che sono lente)
def adaptive_gaussian_distance(x1:torch.tensor, x2:torch.tensor, p:int=2)->torch.Tensor:
    """
    Compute a normalized distance with the (average) sample variance
    Args:
        x1, x2 (torch.Tensor): Tensor di forma (batch_size, D).
    Returns:
        torch.Tensor: Matrice (batch_size, batch_size) con le distanze normalizzate.
    """
    # "empirical" sample variance
    sigma = torch.std(x1, dim=0, unbiased=True, keepdim=True) 
    ## potrei utilizare anche la varianza dell'unione 
    # torch.cat([x1, x2], dim=0)...
    ### ... o fare la media delle varianzae
    #sigma2 = torch.std(x2, dim=0, unbiased=True, keepdim=True).mean()
    #sigma = (sigma1 + sigma2) / 2  # Media delle deviazioni standard
    sigma=sigma.mean()
    ## check https://runebook.dev/en/articles/pytorch/generated/torch.cdist
    dists = torch.cdist(x1, x2, p=p)
    #  Normalizzazione Gaussiana
    return torch.exp(- (dists**2) / (2 * sigma**2))  #


def direction_distance(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
    """
    Qualitative distance just return 0 or 1 according to elments being equal
    or differen
    """
    return (l1[:, None] == l2[None, :]).int()

### periodic distance created in accordance with the  movement
### i.e. hand movement in different directions
def circular_distance(l1: torch.Tensor, l2: torch.Tensor, num_directions=8) -> torch.Tensor:
    """Compute circular distance in a periodic space (modulo num_directions)."""
    # Reshape per broadcasting (batch_size, 1)
    l1 = l1[:, None]  
    l2 = l2[None, :] 
    
    return torch.minimum(torch.abs(l1 - l2), num_directions - torch.abs(l1 - l2))


def minkowski_distance(x1: torch.Tensor, x2: torch.Tensor, p: int = 2, normalize=True) -> torch.Tensor:
    """
    Compute a normalized minkowski distance
    
    Args:
        x1, x2 (torch.Tensor): Tensors di of shape (batch_size, D).
        p (int): Minkowski's distance order. 2 is Default  (Euclidean distance).
        
    Returns:
        torch.Tensor: Matrix (batch_size, batch_size) with normalized distances
    """
    # Calcolo della media e della deviazione standard lungo ogni dimensione
    if normalize:
        mean_x1 = torch.mean(x1, dim=0, keepdim=True)
        std_x1 = torch.std(x1, dim=0, unbiased=True, keepdim=True)
        mean_x2 = torch.mean(x2, dim=0, keepdim=True)
        std_x2 = torch.std(x2, dim=0, unbiased=True, keepdim=True)
        
        # Normalizzazione dei dati
        x1_norm = (x1 - mean_x1) / std_x1
        x2_norm = (x2 - mean_x2) / std_x2
    
        # Calcolo della distanza di Minkowski normalizzata
    dists = torch.cdist(x1_norm, x2_norm, p=p)
    
    return dists