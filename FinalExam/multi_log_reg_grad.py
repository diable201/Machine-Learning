import numpy as np
from scipy.special import expit

def MultiLogRegGrad(X: np.array, y: np.array, W: np.array) -> np.array:
    K = W.shape[1]
    y_onehot = np.eye(K)[y]
    
    grad = X.T.dot(expit(X.dot(W)) - y_onehot) 

    return grad