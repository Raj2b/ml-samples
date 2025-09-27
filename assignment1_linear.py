# Question 1. a) 
# This function will take an n x d input matrix X and an n x 1 target/label vector y, and return a d x 1 vector of 
# weights/parameters w corresponding to the solution of the L2 losses
def minimizeL2(X, y):
    """
    Least-Square (L2) linear regresssion

    Parameters:
        - X: n x d matrix
        - Y: Target vector

    Returns:
        - d x 1 vector of weights/parameters w corresponding to the solution of the L2 losses
    
    
    
    """



    X = np.asarray(X, dtype = float)
    y = np.asarray(y, dtype = float)

    if y.ndim == 1:
        y = y[:, None]  # (n, ) -> (n, 1)
    
    # compute A = X^T * X
    A = X.T @ X 

    # compute b = X^T * y
    b = X.T @ y 
    
    # Solve the linear system Aw = b for w to get the least-squares solution 
    w = np.linalg.solve(A, b)
    return w
