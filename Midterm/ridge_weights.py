import numpy as np

def RidgeWeights(X: np.array, y: np.array, C: float = 1.0) -> np.array:
    num_features = X.shape[1]
    
    identity_matrix = np.identity(num_features)
    
    w = np.linalg.inv(np.dot(X.T, X) + C * identity_matrix).dot(X.T).dot(y)
    
    return w
