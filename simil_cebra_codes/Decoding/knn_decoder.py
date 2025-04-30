from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import numpy as np
import sklearn.metrics
import re
from model_utils import *
from model_utils import run_model
#from process_utils import *



def find_best_k(emb_train, emb_valid, label_train, label_valid):
    metric = 'cosine'
    # Possible values of k
    neighbors = np.power(np.arange(1, 6), 2)
    ## inizializzazione scores e altri valori
    best_score = float('-inf')
    best_k = None
    best_pos_err = None
    best_pos_score = None
    
    for n in neighbors:
        pos_decoder = KNeighborsRegressor(n_neighbors=n, metric=metric)
        dir_decoder = KNeighborsClassifier(n_neighbors=n, metric=metric)
        
        pos_decoder.fit(emb_train, label_train[:, 0])
        dir_decoder.fit(emb_train, label_train[:, 1])
        
        pos_pred = pos_decoder.predict(emb_valid)
        dir_pred = dir_decoder.predict(emb_valid)
        
        prediction = np.stack([pos_pred, dir_pred], axis=1)
        valid_score = sklearn.metrics.r2_score(label_valid[:, :2], prediction)
        pos_err = np.median(np.abs(pos_pred - label_valid[:, 0]))
        pos_score = sklearn.metrics.r2_score(label_valid[:, 0], pos_pred)
        
        if valid_score > best_score:
            best_score = valid_score
            best_k = n
            best_pos_err = pos_err
            best_pos_score = pos_score
    
    return best_k, best_score, best_pos_err, best_pos_score

def decoding_mod(pos_decoder, dir_decoder, emb_test, label_test):
    """
    Use pre-trained decoders to predict and evaluate on the test set.
    """
    pos_pred = pos_decoder.predict(emb_test)
    dir_pred = dir_decoder.predict(emb_test)
    
    prediction = np.stack([pos_pred, dir_pred], axis=1)
    test_score = sklearn.metrics.r2_score(label_test[:, :2], prediction)
    pos_test_err = np.median(np.abs(prediction[:, 0] - label_test[:, 0]))
    pos_test_score = sklearn.metrics.r2_score(label_test[:, 0], prediction[:, 0])
    
    return test_score, pos_test_err, pos_test_score, prediction
