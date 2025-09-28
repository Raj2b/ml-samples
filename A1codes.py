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

def synRegExperiments():

    def genData(n_points, is_training=False):
        '''
        This function generate synthetic data
        '''
        X = np.random.randn(n_points, d) # input matrix
        X = np.concatenate((np.ones((n_points, 1)), X), axis=1) # augment input
        y = X @ w_true + np.random.randn(n_points, 1) * noise # ground truth label
        if is_training:
            y[0] *= -0.1
        return X, y
    n_runs = 100
    n_train = 30
    n_test = 1000
    d = 5
    noise = 0.2
    train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(101269419)

    def L2_loss(X, w, y):
        r = X @ w - y
        return float(0.5 * np.mean(r**2))      # ½·MSE
    def Linf_loss(X, w, y):
        return float(np.max(np.abs(X @ w - y)))

    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)

        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)

        train_loss[r, 0, 0] = L2_loss(Xtrain,  w_L2,   ytrain)   # L2 model,  L2 metric
        train_loss[r, 0, 1] = Linf_loss(Xtrain, w_L2,   ytrain)  # L2 model,  Linf metric
        train_loss[r, 1, 0] = L2_loss(Xtrain,  w_Linf, ytrain)   # Linf model, L2 metric
        train_loss[r, 1, 1] = Linf_loss(Xtrain, w_Linf, ytrain)  # Linf model, Linf metric

        # store test losses
        test_loss[r, 0, 0] = L2_loss(Xtest,  w_L2,   ytest)
        test_loss[r, 0, 1] = Linf_loss(Xtest, w_L2,   ytest)
        test_loss[r, 1, 0] = L2_loss(Xtest,  w_Linf, ytest)
        test_loss[r, 1, 1] = Linf_loss(Xtest, w_Linf, ytest)

        # ----- averages over runs -> 2×2 tables -----
    train_avg = train_loss.mean(axis=0)
    test_avg  = test_loss.mean(axis=0)
    return train_avg, test_avg