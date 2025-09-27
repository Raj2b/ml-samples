import numpy as np

def minimuizeL2(X, y):
    return np.linalg.solve(X.T @ X, X @ y)

