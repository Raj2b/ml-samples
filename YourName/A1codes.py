import numpy as np
from cvxopt import matrix, solvers
from scipy.optimize import minimize


#1.a)
def minimizeL2(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


#1.b)
def minimizeLinf(X, y):
    # assume X is (n,d) and y is (n,1)...
    n, d = X.shape

    # build objective vector: c^T u = δ
    # u = [w; δ] in R^(d+1)
    c = np.concatenate((np.zeros((d,1)), np.array([[1.0]])), axis=0)

    # constraint 1: make sure delta is not negative  (δ ≥ 0)
    G1 = np.concatenate((np.zeros((1,d)), -np.ones((1,1))), axis=1)

    # constraint 2: predicted values (Xw) cannot be much larger than y  (Xw - y ≤ δ)
    G2 = np.concatenate((X, -np.ones((n,1))), axis=1)

    # constraint 3: predicted values (Xw) cannot be much smaller than y  (y - Xw ≤ δ)
    G3 = np.concatenate((-X, -np.ones((n,1))), axis=1)

    # stacking the constraints together...
    # G will contain all the "rules" we made (rows from G1, G2, G3)
    # h will contain the numbers on the right-hand side of each rule so that the solver knows 
    # to enforce:   G u ≤ h
    G = np.vstack((G1, G2, G3))
    h = np.vstack((np.zeros((1,1)), y, -y))

    # solve LP: minimize c^T u s.t. G u <= h
    solvers.options['show_progress'] = False # just to hide the terminal output to prevent
                                             # spamming

    result = solvers.lp(matrix(c), matrix(G), matrix(h))

    u_star = np.array(result['x'])

    # return the first d entries of u* (we only care about the weight vector w*, not δ*)
    # reshaped as a column vector
    return u_star[:d].reshape(d,1)

'''''
#1.c) NEEDS TO BE CHANGED, FULL CHAT
'''''
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


#2.a.1) FULL CHAT
def linearRegL2Obj(w, X, y):
    w = np.asarray(w).reshape(-1,1)   # accept (d,) or (d,1)
    y = np.asarray(y).reshape(-1,1)
    r = X @ w - y
    n = X.shape[0]
    return float(0.5 / n * (r.T @ r))

def linearRegL2Grad(w, X, y):
    w = np.asarray(w).reshape(-1,1)
    y = np.asarray(y).reshape(-1,1)
    r = X @ w - y
    n = X.shape[0]
    return (X.T @ r) / n  

#2.a.2) FUL CHAT

def find_opt(obj_func, grad_func, X, y):
    d = X.shape[1]
    w0 = np.random.randn(d)           

    def func(w): 
        return float(obj_func(w, X, y))      
    
    def gd(w):   
        return grad_func(w, X, y).ravel()    

    res = minimize(func, w0, jac=gd, method="BFGS")
    return res.x.reshape(-1,1)        

#2.b) 
def logisticRegObj(w, X, y):
    # Accept (d,) or (d,1); return scalar
    w = np.asarray(w).reshape(-1,1)
    y = np.asarray(y).reshape(-1,1)
    z = X @ w
    # Use logaddexp for a stable CE:  -y*logσ(z) - (1-y)*log(1-σ(z)) = softplus(z) - y*z
    # softplus(z) = log(1+exp(z)) implemented stably by np.logaddexp(0, z)
    loss = np.logaddexp(0.0, z) - y * z
    return float(loss.mean())


def logisticRegGrad(w, X, y):
    # ∇J(w) = (1/n) X^T (σ(Xw) - y)
    w = np.asarray(w).reshape(-1,1)
    y = np.asarray(y).reshape(-1,1)
    z = X @ w
    p = 1.0 / (1.0 + np.exp(-z))     # sigmoid
    return (X.T @ (p - y)) / X.shape[0]