import numpy as np

def zscore(X):
    X = X - np.mean(X,axis=0,keepdims=True)
    X = X / np.std(X,axis=0,keepdims=True)
    return X