import numpy as np
from cvxopt import matrix, solvers

def minimizeL2(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)



def minimizeLinf(X, y):
    X = np.asarray(X, float)
    y = np.asarray(y, float).reshape(-1, 1)
    n, d = X.shape

    c  = np.vstack([np.zeros((d,1)), np.ones((1,1))])       # minimize δ
    G1 = np.hstack([np.zeros((1,d)), -np.ones((1,1))])      # -δ ≤ 0
    G2 = np.hstack([ X, -np.ones((n,1))])                   # Xw - y ≤ δ·1  → [X, -1]u ≤ y
    G3 = np.hstack([-X, -np.ones((n,1))])                   # y - Xw ≤ δ·1  → [-X, -1]u ≤ -y  ✅

    G = np.vstack([G1, G2, G3])
    h = np.vstack([np.zeros((1,1)), y, -y])                 # note the -y here  ✅

    solvers.options['show_progress'] = False
    u = np.array(solvers.lp(matrix(c), matrix(G), matrix(h))['x'])
    return u[:d].reshape(d,1)