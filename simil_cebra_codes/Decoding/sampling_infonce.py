# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:56:59 2024

@author: zlollo2
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

### infonce calculation
def infonce(pos_dist, neg_dist):
    """
    Calculate the InfoNCE loss from positive and negative distances (inputs)
	Gives Loss (Output)
	
    """
    neg_logsumexp = torch.logsumexp(neg_dist, dim=1)
    loss = -torch.mean(pos_dist - neg_logsumexp)
    return loss

#### metric is cosine  similarity
def dot_similarity(ref, pos, neg):
    """ Compute the dot product similarity for reference, positive, and negative samples. """
    # Positive similarity
    pos_sim = torch.einsum('nd,nd->n', ref, pos) 

    # Negative similarities across all negatives
    neg_sim = torch.einsum('nd,nmd->nm', ref, neg) 
    return pos_sim, neg_sim

## Esempio
 #Campioni di riferimento e positivi di forma (N, D)
#ref = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
#pos = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.float32)
#ref=ref_samples
#neg=neg_samples
#pos=pos_samples_after
# # # Campioni negativi di forma (M, D)
#neg = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float32)

# # Calcolo delle similarità
#pos_sim, neg_sim = dot_similarity(ref, pos, neg)

#print("Similarità Positiva:", pos_sim)
#print("Similarità Negativa:", neg_sim)


def compute_contrastive_loss(ref_samples, pos_samples, neg_samples):
    """
    Compute the contrastive loss using InfoNCE based on cosine similarities.
    """
    
    
    ref_samples=check_and_normalize(ref_samples)
    neg_samples=check_and_normalize(neg_samples)
    pos_samples=check_and_normalize(pos_samples)

    
    pos_dist, neg_dist = dot_similarity(ref_samples, pos_samples, neg_samples)
    
    loss = infonce(pos_dist, neg_dist)
    return loss

def check_and_normalize(tensor):
    """
    Check if tensor length is one and eventually normalize it
    
    Gets a tensor as input, receive a tensor (nromalized)
    
    """
    # Comopute feature norm along cols 
    norm = torch.norm(tensor, dim=-1, keepdim=True)
    norm_1 = torch.tensor(1.0, dtype=norm.dtype, device=norm.device)
    # if norm not (close to) 1, nomralize
    if not torch.allclose(norm, norm_1, atol=1e-6):
        # Normalizza il tensore se la norma non è circa 1
        tensor = tensor / norm
    return tensor


def sample_time_contrastive_data(data, time_offset, n_anchor=1, n_neg=1, bidirectional=False):
    """
    Sample reference, positive, and negative examples for time-contrastive 
    learning.
    
    Args:
        data (torch.Tensor or np.ndarray): Input data of shape (T, n), where 
        T is the number of time steps and n is the number of features.
        
        n_anchor (int): Number of reference samples to draw.(default=1)
        time_offset (int): Temporal offset used to sample positive examples.
        n_neg (int): Number of negative samples to draw for each reference 
        sample.
        bidirectional (bool): Whether to sample positive examples before and 
        after the reference.
    
    Returns:
        Tuple of reference samples, positive samples after, (positive samples
                               before if bidirectional), negative samples.
    """
        
    is_numpy = isinstance(data, np.ndarray)
    if is_numpy:
        data = torch.from_numpy(data)
    
    # Check if time_offset is valid given the data size
    assert len(data) >= n_anchor + 2 * time_offset + 1 + n_neg, \
        f"Data length too short for sampling must be at least n_anchor+2*time_offset+1+n_neg"
    
    # Define the range of valid reference indices
    max_ref_idx = data.size(0) - time_offset
    min_ref_idx = time_offset if bidirectional else 0
    
    # Sample reference indices randomly within the valid range
    ## add min ref to fall within the range
    ref_indices = torch.randperm(max_ref_idx - min_ref_idx)[:n_anchor] + min_ref_idx
    
    # Ensure that reference indices are valid within data range
    # ref_indices = ref_indices[ref_indices < data.size(0)]
    
    # Sample positive indices after the reference
    pos_indices = ref_indices + time_offset
    #pos_indices_after = pos_indices_after[pos_indices_after < data.size(0)]
    
    # Sample positive indices before the reference (if bidirectional)
    # in caso se decido di campionare bidirezionalmente
    pos_indices_before = ref_indices - time_offset if bidirectional else None
    #if bidirectional:
    #   pos_indices_before = pos_indices_before[pos_indices_before >= 0]
    
    # Initialize tensor for negative indices
    neg_indices = torch.zeros((n_anchor, n_neg), dtype=torch.long)
    
    # Sample negative indices ensuring they are at least time_offset away from reference indices
    for i in range(n_neg):
        while True:
            neg_idx = torch.randint(0, data.size(0), (n_anchor,))
            cond_after = torch.abs(neg_idx - ref_indices) > time_offset
            cond_before = torch.abs(neg_idx - ref_indices) > time_offset if bidirectional else torch.tensor([True])
            if torch.all(cond_after & cond_before):
                neg_indices[:, i] = neg_idx
                break
    
    # Extract samples based on the indices
    ref_samples = data[ref_indices]
    pos_samples = data[pos_indices]
    pos_samples_before = data[pos_indices_before] if bidirectional else None
    neg_samples = data[neg_indices]
    
    # Return the appropriate samples depending on bidirectional flag
    if bidirectional:
            return ref_samples, pos_samples, neg_samples,pos_samples_before,ref_indices, pos_indices,  neg_indices, pos_indices_before
    else:
            return ref_samples, pos_samples, neg_samples,ref_indices, pos_indices,  neg_indices


# ref_,pos_,neg_, _,_,_=sample_time_contrastive_data(data_,5, 1,8)

# pos_.numpy()


# norm = torch.norm(neg_, dim=-1, keepdim=True)
# norm.numpy()
# neg_=check_and_normalize(neg_)

# loss=compute_contrastive_loss(ref_,pos_, neg_).numpy()
# neg_.numpy()
# assert norm.any().numpy()==1
# p.numpy()

# # # #############
# # T, N = 100, 10
# # data_ = np.random.randn(T, N)
   
# # #sample_time_contrastive_data(data, time_offset, n_anchor=1, n_neg=1,  bidirectional=False):
# # ref_samples, pos_samples_after, neg_samples, ref_indices, pos_indices_after, neg_indices=sample_time_contrastive_data(data_, 7,4,1,10)

# # compute_contrastive_loss(ref_samples,pos_samples_after,neg_samples)


# # def main():
# #     # Esempio di dati con dimensione T*N (T: timesteps, N: features)
# #     T, N = 100, 10
# #     data = np.random.randn(T, N)
    
# #     # Parametri per il campionamento
# #     num_ref_samples = 20
# #     time_offset = 5
# #     num_neg_samples = 5
# #     bidirectional = False  # Se True, considera anche campioni positivi prima del riferimento

# #     # Campionamento dei dati
# #     ref_samples, pos_samples_after, neg_samples = sample_time_contrastive_data(data, num_ref_samples, time_offset, num_neg_samples, bidirectional)

# #     # Calcolo della loss contrastiva
# #     loss = compute_contrastive_loss(ref_samples, pos_samples_after, neg_samples)
    
# #     # Stampa della perdita
#     print(f'Loss: {loss.item()}')

# if __name__ == '__main__':
    
#     main()
    

    