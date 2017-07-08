import numpy as np

def rand_proj(X,d):
    G_rows = X.shape[1]
    G_cols = d
    G = np.random.normal(0,1,(G_rows,G_cols))
    X1 = np.dot(X,G)
    return X1
