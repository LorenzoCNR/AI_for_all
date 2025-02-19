# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:33 2025

@author: loren
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Simuliamo un embedding GPT-2 per una frase con 5 parole (ogni parola ha un vettore di 300 dimensioni)
num_words = 5
embedding_dim = 300

# Generiamo un embedding casuale (in una situazione reale, verrebbe da GPT-2)
np.random.seed(42)
word_embeddings = np.random.randn(num_words, embedding_dim)  

# Simuliamo l'attività cerebrale corrispondente dello speaker e del listener
# (Ogni "neurone" risponde con un'attività diversa a ciascun embedding)
num_neurons = 50

# Generiamo matrici di attività neurale come combinazione lineare degli embedding + rumore
speaker_activity = word_embeddings @ np.random.randn(embedding_dim, num_neurons) + np.random.randn(num_words, num_neurons) * 0.1
listener_activity = np.roll(speaker_activity, shift=1, axis=0)  # Il listener ha un ritardo nella risposta

# Addestriamo un modello di regressione per prevedere l'attività cerebrale dello speaker dagli embedding
ridge_speaker = Ridge(alpha=1.0)
ridge_speaker.fit(word_embeddings, speaker_activity)

# Addestriamo un modello per prevedere l'attività cerebrale del listener dagli embedding
ridge_listener = Ridge(alpha=1.0)
ridge_listener.fit(word_embeddings, listener_activity)

# Valutiamo la capacità predittiva
speaker_pred = ridge_speaker.predict(word_embeddings)
listener_pred = ridge_listener.predict(word_embeddings)

# Plot per confrontare le attività reali e predette
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(speaker_activity[:, 0], label="Speaker Actual")
plt.plot(speaker_pred[:, 0], '--', label="Speaker Predicted")
plt.title("Encoding Model: Speaker")
plt.xlabel("Word index")
plt.ylabel("Neural Activity")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(listener_activity[:, 0], label="Listener Actual")
plt.plot(listener_pred[:, 0], '--', label="Listener Predicted")
plt.title("Encoding Model: Listener")
plt.xlabel("Word index")
plt.ylabel("Neural Activity")
plt.legend()

plt.show()
