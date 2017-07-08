import numpy as np
class MyLogisticReg2:
    
    w = 0
    n_classes = 0
    n_samples = 0
    n_features = 0
    nIter = 0
    
    def __init__(self):
        self.w = 0
        
        
    def fit(self, X_train, r):
        X_train = np.divide((X_train - np.min(X_train,axis=0)),(np.max(X_train, axis=0) - np.min(X_train,axis=0)))
        lr = 0.001
        nIter = 1000
        n_samples, n_features = X_train.shape
        X_train = np.concatenate((X_train, np.ones(n_samples)[np.newaxis].T), axis = 1)
        n_features = n_features + 1       
        self.w = (np.random.rand(n_features) - 0.5)*0.05
        
              
        for i in range(nIter):
            lin_score = np.matmul(X_train, self.w)
            lin_score_max = np.abs(np.max(lin_score))
            #if lin_score_max >= 800:
            #    lin_score = lin_score/lin_score_max
            y_sig = np.exp(lin_score)/(1+np.exp(lin_score))
            delW = np.matmul(X_train.T, (y_sig-r))
            self.w = self.w - lr*delW
        
    def predict(self, X_test):
        X_test = np.divide((X_test - np.min(X_test,axis=0)),(np.max(X_test, axis=0) - np.min(X_test,axis=0)))
        n_samples, n_features = X_test.shape
        X_test = np.concatenate((X_test, np.ones(n_samples)[np.newaxis].T), axis = 1)
        n_features = n_features + 1
        y_pred = np.zeros(n_samples)
        lin_score = np.zeros(n_samples)
        for i in range(0,n_samples):
            lin_score[i] = np.matmul(self.w.T, X_test[i, :])
            #score_pred[i] = 1/(1+np.exp(-lin_score[i]))
            
        #y_pred = np.array(score_pred > 0.5, dtype = int) 
        y_pred[lin_score > 0] = 1
        return y_pred