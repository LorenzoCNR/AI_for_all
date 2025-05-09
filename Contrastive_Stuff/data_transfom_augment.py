# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 10:41:03 2025

@author: loren
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from scipy.stats import skew

import numpy as np
import pandas as pd
from scipy.stats import skew

# 1️⃣ **Padding per differenziazione prima**
def first_difference(signal, c_t):
    
    """
    
    Calcola la differenza prima con padding iniziale.
    c_t indica inizio e fine di ogni trial
    lunghezza dei trial è la medesima per tutti i trial o una lista 
    o vettore np con le lunghezze di ogni trial
    """
    result=[]
    
    is_1d=len(signal.shape)==1
    for i in range(len(c_t)-1):
        print(c_t[i],c_t[i+1])
        trial=signal[c_t[i]:c_t[i+1]]
        diff = np.diff(trial, n=1, axis=0)
        pad = np.zeros((1,))
       
        if is_1d:
            # Se 1D, un solo zero
            pad = np.zeros((1,))  
            trial_diff = np.concatenate([pad, diff])

        else:
            # Se 2D, zero per ogni canale
            pad = np.zeros((1, trial.shape[1]))  
            trial_diff = np.vstack([pad, diff])
        result.append(trial_diff)
    return result


def decompose_signal(signal, c_t, sampling_rate=1000):
     
    result=[]
    is_1d=len(signal.shape)==1
    for i in range(len(c_t)-1):
         print(c_t[i],c_t[i+1])
         trial=signal[c_t[i]:c_t[i+1]]
         ft_vals = np.fft.fft(trial)
         freqs = np.fft.fftfreq(len(trial), d=1/sampling_rate)
         
         return ft_vals, freqs
         
         dominant_freq = freqs[np.argmax(np.abs(fft_vals[1:])) + 1]
          # Evita divisione per 0
         dominant_period = int(1 / dominant_freq) if dominant_freq != 0 else len(signal) 


ft, ff=decompose_signal(X, c_t)

   return dominant_freq, dominant_period
         diff = np.diff(trial, n=1, axis=0)
         pad = np.zeros((1,))
        
         if is_1d:
            # Se 1D, un solo zero
             pad = np.zeros((1,))  
             trial_diff = np.concatenate([pad, diff])

         else:
            # Se 2D, zero per ogni canale
             pad = np.zeros((1, trial.shape[1]))  
             trial_diff = np.vstack([pad, diff])
         result.append(trial_diff)
     return result

 
signal=X[:,1]
signal[c_t[1]:c_t[2]]
result=[]
pippo=first_difference(X,c_t)

pippo_=pippo.flatten()      


# 2️⃣ **Padding per rolling variance**
def rolling_variance(signal, window_size=50):
    """Calcola la rolling variance con padding iniziale per mantenere la stessa dimensione."""
    rolling_var = pd.DataFrame(signal).rolling(window=window_size, center=False).var().fillna(method="bfill").values
    return rolling_var

# 3️⃣ **Padding per rolling skewness**
def rolling_skewness(signal, window_size=50):
    """Calcola la rolling skewness con padding iniziale."""
    skewness = np.array([skew(signal[i:i+window_size]) if i+window_size < len(signal) else 0 
                         for i in range(len(signal))])
     # Padding replicando il primo valore valido
    pad = np.full(window_size, skewness[window_size]) 
    return np.concatenate([pad, skewness[window_size:]])
# Funzione per il filtraggio passa-banda
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)


for i in range(len(c_t)-2):
    print(c_t[i],c_t[i+1])


X_vol=X**2
X_abs=abs(X)
X_bpf=bandpass_filter(X, 4, 8, 1000)

signal=X[:,1]
signal[c_t[1]:c_t[2]]
result=[]
pippo=first_difference(X,c_t)

pippo_=pippo.flatten()
