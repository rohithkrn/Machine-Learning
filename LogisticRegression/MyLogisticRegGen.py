import numpy as np

class MyLogisticRegGen:
    w = 0
    classes = 0
    count = 0
    n_classes = 0
    n_samples = 0
    n_features = 0
    nIter = 0
    
    def __init__(self):
        self.w = 0
        
        
    def fit(self, X_train, r):
        
        self.classes, self.count = np.unique(r, return_counts=True)
        self.n_classes = np.size(self.classes)
        lr = 0.0001
        nIter = 500
        n_samples, n_features = X_train.shape
        X_train = np.concatenate((X_train, np.ones(n_samples)[np.newaxis].T), axis = 1)
        n_features = n_features + 1
        self.w = (np.random.rand(n_features, self.n_classes) - 0.5)*0.05
        for i in range(0,nIter):
            lin_score = np.matmul(X_train, self.w)
            lin_score_max = np.abs(np.max(lin_score))
            y_sig = np.exp(lin_score)
            yr = np.zeros((y_sig.shape))
            exp_score_sum = np.sum(y_sig, axis=1)
            for j in range(0, n_samples):
                y_sig[j,:] = (y_sig[j,:]/exp_score_sum[j]+1e-6)
                yr[j,r[j]] = 1
            delW = np.matmul(X_train.T, (y_sig-yr))
            self.w = self.w - lr*delW
        
    def predict(self, X_test):

        n_samples, n_features = X_test.shape
        X_test = np.concatenate((X_test, np.ones(n_samples)[np.newaxis].T), axis = 1)
        n_features = n_features + 1
        y_pred = np.zeros(n_samples)
        lin_score = np.matmul(X_test, self.w)
        y_pred = self.classes[lin_score.argmax(axis=1)]
        return y_pred