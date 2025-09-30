import os
import numpy as np
from cvxopt import matrix, solvers
import pandas as pd
from scipy.optimize import minimize


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

    for r in range(n_runs):
        w_true = np.random.randn(d + 1, 1)
        Xtrain, ytrain = genData(n_train, is_training=True)
        Xtest, ytest = genData(n_test, is_training=False)
        w_L2 = minimizeL2(Xtrain, ytrain)
        w_Linf = minimizeLinf(Xtrain, ytrain)
        res = Xtrain @ w_L2 - ytrain
        train_loss[r, 0, 0] = float(0.5 * np.mean(res**2))         #L2 loss
        train_loss[r, 0, 1] = float(np.max(np.abs(res)))           #Linf loss
        # Linf model
        res = Xtrain @ w_Linf - ytrain
        train_loss[r, 1, 0] = float(0.5 * np.mean(res**2))         #L2 loss
        train_loss[r, 1, 1] = float(np.max(np.abs(res)))           #Linf loss

        #TEST LOSSES
        # L2 model
        res = Xtest @ w_L2 - ytest
        test_loss[r, 0, 0] = float(0.5 * np.mean(res**2))          #L2 loss
        test_loss[r, 0, 1] = float(np.max(np.abs(res)))            #Linf loss
        # Linf model
        res = Xtest @ w_Linf - ytest
        test_loss[r, 1, 0] = float(0.5 * np.mean(res**2))          #L2 loss
        test_loss[r, 1, 1] = float(np.max(np.abs(res)))            #Linf loss

    #compute the average losses over runs
    train_loss = train_loss.mean(axis=0)  # -> (2, 2)
    test_loss  = test_loss.mean(axis=0)   # -> (2, 2)

    #return a 2-by-2 training loss variable and a 2-by-2 test loss variable
    return train_loss, test_loss


# PART 1.D.1)
def preprocessCCS(dataset_folder):

    # build path to the Excel file
    path = os.path.join(dataset_folder, "Concrete_Data.xls")

    # read dataset using pandas
    df = pd.read_excel(path, engine="xlrd")

    # X = features (all but last column)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    
    # target (last column) as (n,1)
    y = df.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)

    return X, y

# PART 1.D.2)
def L2_loss(X, w, y):
    r = X @ w - y
    return float(0.5 * np.mean(r**2)) # 1/2 scaling factor

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
        tr, te = idx[:n//2], idx[n//2:] 
        Xtr, ytr = X[tr], y[tr] # Xtr is X (features) for the training set,
        Xte, yte = X[te], y[te] # Xte is X (features) for the test set
                                # ytr is y (targets) for the training set
                                # yte is y (targets) for the test set

        # TODO: Learn two different models from the training data
        # using L2 and L infinity losses
        w_L2   = minimizeL2(Xtr, ytr) # solves ordinary least square problem
        w_Linf = minimizeLinf(Xtr, ytr) # solves a linear program to minimize the 
                                        # max absolute error

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

#2.a.1) FULL CHAT
def linearRegL2Obj(w, X, y):
    w = np.asarray(w).reshape(-1,1)   #ensure w is (d,1) column vector that accepts (d,) or (d, 1) input
    y = np.asarray(y).reshape(-1,1)   #ensure y is (n,1)
    r = X @ w - y                     #residuals
    n = X.shape[0]                    
    return float(0.5 / n * (r.T @ r)) #objective J(w)

def linearRegL2Grad(w, X, y):
    w = np.asarray(w).reshape(-1,1)  #ensure w is (d,1)
    y = np.asarray(y).reshape(-1,1)  #y is (n,1)
    r = X @ w - y                       
    n = X.shape[0]
    return (X.T @ r) / n        #gradient ∇J(w) 

#2.a.2) FUL CHAT

def find_opt(obj_func, grad_func, X, y):
    d = X.shape[1]
    w_0 = np.random.randn(d)     #initialize random 1-D array      

    #Define an objective function `func` that takes a single argument (w)
    def func(w): 
        return float(obj_func(w, X, y))      
    
    #Define a gradient function `gd` that takes a single argument (w)
    def gd(w):   
        return grad_func(w, X, y).ravel()    

    return minimize(func, w_0, jac=gd)['x'][:, None]   
       

#2.b) 
def logisticRegObj(w, X, y):
    # Accept (d,) or (d,1)
    w = np.asarray(w).reshape(-1,1)
    y = np.asarray(y).reshape(-1,1)
    z = X @ w
    # Use logaddexp for a stable CE:  -y*logσ(z) - (1-y)*log(1-σ(z)) = log(1 + exp(z)) - y*z
    loss = np.logaddexp(0.0, z) - y * z
    #return avg loss over all examples 
    return float(loss.mean())


def logisticRegGrad(w, X, y):
    # ∇J(w) = (1/n) X^T (σ(Xw) - y)
    #shapes to columns
    w = np.asarray(w).reshape(-1,1)
    y = np.asarray(y).reshape(-1,1)
    #scores, z = Xw with shape (n,1)
    z = X @ w

    #predicted probabilites, clip z to avoid overflow when z is large
    p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
    #avg grazdient over n examples
    return (X.T @ (p - y)) / X.shape[0]

#2.c.1)
def synClsExperiments():

    def genData(n_points, dim1, dim2):
        '''
        This function generate synthetic data
        '''
        c0 = np.ones([1, dim1]) # class 0 center
        c1 = -np.ones([1, dim1]) # class 1 center
        X0 = np.random.randn(n_points, dim1 + dim2) # class 0 input
        X0[:, :dim1] += c0
        X1 = np.random.randn(n_points, dim1 + dim2) # class 1 input
        X1[:, :dim1] += c1
        X = np.concatenate((X0, X1), axis=0)
        X = np.concatenate((np.ones((2 * n_points, 1)), X), axis=1) # augmentation
        y = np.concatenate([np.zeros([n_points, 1]), np.ones([n_points, 1])], axis=0)
        return X, y
    
    def runClsExp(m=100, dim1=2, dim2=2):
        '''
        Run classification experiment with the specified arguments
        '''
        n_test = 1000
        Xtrain, ytrain = genData(m, dim1, dim2)
        Xtest, ytest = genData(n_test, dim1, dim2)
        w_logit = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)
        ytrain_hat = ((Xtrain @ w_logit) >= 0).astype(float) #Compute predicted labels of the training points
        train_acc = float((ytrain_hat == ytrain).mean()) #Compute the accuarcy of the training set
        ytest_hat = ((Xtest  @ w_logit) >= 0).astype(float) #Compute predicted labels of the test points
        test_acc = float((ytest_hat == ytest).mean()) #Compute the accuarcy of the test set  

        return train_acc, test_acc

    n_runs = 100
    train_acc = np.zeros([n_runs, 4, 3])
    test_acc = np.zeros([n_runs, 4, 3])
    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(101269419)
    for r in range(n_runs):
        for i, m in enumerate((10, 50, 100, 200)):
            train_acc[r, i, 0], test_acc[r, i, 0] = runClsExp(m=m)
        for i, dim1 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 1], test_acc[r, i, 1] = runClsExp(dim1=dim1)
        for i, dim2 in enumerate((1, 2, 4, 8)):
            train_acc[r, i, 2], test_acc[r, i, 2] = runClsExp(dim2=dim2)

    #compute the average accuracies over runs
    train_acc = train_acc.mean(axis=0)
    test_acc  = test_acc.mean(axis=0)

    #return a 4-by-3 training accuracy variable and a 4-by-3 test accuracy variable
    return train_acc, test_acc

#2.d) 
def preprocessBCW(dataset_folder):
    path = os.path.join(dataset_folder, "wdbc.data")

    df = pd.read_csv(path, header=None) # to avoid the first row not being read error

    # mapping the labels and removing the ID column...
    y = df.iloc[:, 1].map({'B': 0, 'M': 1}).to_numpy(dtype=float).reshape(-1, 1)
    X = df.iloc[:, 2:].to_numpy(dtype=float)

    return X, y

def runBCW(dataset_folder):
    X, y = preprocessBCW(dataset_folder)
    n, d = X.shape
    X = np.concatenate((np.ones((n, 1)), X), axis=1) # augment

    n_runs = 100
    train_acc = np.zeros([n_runs])
    test_acc = np.zeros([n_runs])

    # TODO: Change the following random seed to one of your student IDs
    np.random.seed(101269165)

    for r in range(n_runs):
        # TODO: Randomly partition the dataset into two parts (50%
        # training and 50% test)
        idx = np.random.permutation(n)
        tr, te = idx[:n//2], idx[n//2:]

        # slice features/labels using index split defined above
        Xtrain, ytrain = X[tr], y[tr]   # training subset
        Xtest,  ytest  = X[te], y[te]   # test subset

        w = find_opt(logisticRegObj, logisticRegGrad, Xtrain, ytrain)

        # TODO: Evaluate the model's accuracy on the training
        # data. Save it to `train_acc`
        yhat_train = (Xtrain @ w >= 0).astype(float) # predict labels: score >= 0 → 1, else 0
        # accuracy = fraction of the correct matches
        train_acc[r] = float((yhat_train == ytrain).mean())

        # TODO: Evaluate the model's accuracy on the test
        # data. Save it to `test_acc`

        # same ideas as above but with the test data this time...
        yhat_test = (Xtest @ w >= 0).astype(float)
        test_acc[r] = float((yhat_test == ytest).mean())

    # TODO: compute the average accuracies over runs
    train_avg = float(train_acc.mean())
    test_avg  = float(test_acc.mean())

    # TODO: return two variables: the average training accuracy and average test accuracy
    return train_avg, test_avg