import numpy as np

def quad_proj(X):

    s1,s2 = X.shape
    X2 = np.concatenate((X,np.square(X)),axis=1)

    for ii in range(s2-1):
        X_inter = np.zeros((s1,s2-ii-1))
        for jj in range(s1):
            X_inter[jj,:] = np.array(X[jj,ii]*X[jj,ii+1:])
        X2 = np.concatenate((X2,X_inter),axis=1)
    
    return X2
