import os
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

#1. a)
def minimizeL2(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)


#1. b)
# USE THIS ONE AS IT HAS BEEN REFACTORED 
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

#1. c)
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


def preprocessCCS(dataset_folder):
    """
    Load the Concrete Compressive Strength dataset.
    
    Args:
        dataset_folder (str): absolute path to folder containing Concrete_Data.xls

    Returns:
        X : (n, d) numpy array of features
        y : (n, 1) numpy array of targets
    """
    # build path to the Excel file
    path = os.path.join(dataset_folder, "Concrete_Data.xls")

    # read dataset (pandas automatically parses the xls)
    df = pd.read_excel(path, engine="xlrd")

    # features = all columns except the last, target = last column
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    # leave bias column out here — add it later in runCCS
    return X, y


import os
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd

def minimizeL2(X, y):
    return np.linalg.solve(X.T @ X, X.T @ y)



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


def preprocessCCS(dataset_folder):
    """
    Load the Concrete Compressive Strength dataset.
    
    Args:
        dataset_folder (str): absolute path to folder containing Concrete_Data.xls

    Returns:
        X : (n, d) numpy array of features
        y : (n, 1) numpy array of targets
    """
    # build path to the Excel file
    path = os.path.join(dataset_folder, "Concrete_Data.xls")

    # read dataset (pandas automatically parses the xls)
    df = pd.read_excel(path, engine="xlrd")

    # features = all columns except the last, target = last column
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    # leave bias column out here — add it later in runCCS
    return X, y

def L2_loss(X, w, y):
    r = X @ w - y
    return float(np.mean(r**2))  # MSE (no 1/2)

def Linf_loss(X, w, y):
    return float(np.max(np.abs(X @ w - y)))  # L_infinity (max abs error)


def runCCS(dataset_folder):
    X, y = preprocessCCS(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

    n_runs = 100
    train_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics
    test_loss = np.zeros([n_runs, 2, 2]) # n_runs * n_models * n_metrics

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(101269165)

    for r in range(n_runs):
        # TODO: Randomly partition the dataset into two parts (50%
        # training and 50% test)
        idx = np.random.permutation(n)
        split = n // 2
        tr, te = idx[:split], idx[split:]
        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        # TODO: Learn two different models from the training data
        # using L2 and L infinity losses
        w_L2   = minimizeL2(Xtr, ytr)
        w_Linf = minimizeLinf(Xtr, ytr)

        # TODO: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the training
        # data). Save them to `train_loss`
        train_loss[r, 0, 0] = L2_loss(Xtr,  w_L2,   ytr)   # L2 model,   L2 metric
        train_loss[r, 0, 1] = Linf_loss(Xtr, w_L2,   ytr)  # L2 model,   L_inf metric
        train_loss[r, 1, 0] = L2_loss(Xtr,  w_Linf, ytr)   # L_inf model, L2 metric
        train_loss[r, 1, 1] = Linf_loss(Xtr, w_Linf, ytr)  # L_inf model, L_inf metric

        # TODO: Evaluate the two models' performance (for each model,
        # calculate the L2 and L infinity losses on the test
        # data). Save them to `test_loss`
        test_loss[r, 0, 0] = L2_loss(Xte,  w_L2,   yte)
        test_loss[r, 0, 1] = Linf_loss(Xte, w_L2,   yte)
        test_loss[r, 1, 0] = L2_loss(Xte,  w_Linf, yte)
        test_loss[r, 1, 1] = Linf_loss(Xte, w_Linf, yte)

        # TODO: compute the average losses over runs
        train_avg = train_loss.mean(axis=0)
        test_avg  = test_loss.mean(axis=0)

        # TODO: return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return train_avg, test_avg