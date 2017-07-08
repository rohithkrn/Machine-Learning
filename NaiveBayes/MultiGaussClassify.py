# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:35:46 2017

@author: rohit
"""
#%%
import numpy as np

#%%
class MultiGaussClassify():
    
    # Data features
    classes = 0
    count = 0 
    n_classes = 0 
    n_samples = 0
    n_features = 0

    # Model parameters
    prior_prob = 0
    class_mean = 0
    covar_mat = 0
    
    error = 0

    def __init__(self):

        self.prior_prob = np.zeros(self.n_classes)
  

    ## Fit the model             
        
    def fit(self,X_train,y_train):
        
        self.classes,self.count = np.unique(y_train,return_counts=True)
        self.n_classes = len(self.classes)
        self.n_samples, self.n_features = np.shape(X_train)
        
        strat_classes = [np.zeros([self.count[i],self.n_features]) for i in range(self.n_classes)]            
   
        ind_arr = np.zeros(self.n_classes,dtype = int) 
    
        for i in range(self.n_samples):
            for j in range(self.n_classes):
                if y_train[i] == self.classes[j]:
                    ind = ind_arr[j]
                    strat_classes[j][ind] = X_train[i]
                    ind_arr[j] += 1
                    break   
        
## Computing model parameters  
        
        self.prior_prob = np.zeros(self.n_classes)
        self.class_mean = [np.zeros(self.n_features) for i in range(self.n_classes)] 
        class_var = np.zeros([self.n_features,self.n_features])
        self.covar_mat = [np.zeros([self.n_features,self.n_features]) for i in range(self.n_classes)]
        
        for ii in range(self.n_classes):
            self.prior_prob[ii] = self.count[ii]/self.n_samples
            self.class_mean[ii] = np.mean(strat_classes[ii],axis = 0) 
            mean_cent = strat_classes[ii] - self.class_mean[ii]
            class_var = np.matmul(np.transpose(mean_cent),mean_cent)
            class_var = class_var/len(strat_classes[ii])
            self.covar_mat[ii] = class_var    
                   
    ## Prdiction    
        
    def predict(self,X_test):

        s1,s2 = X_test.shape
        y_pred = np.zeros(s1)

        detS = np.zeros(self.n_classes)
        logDetS = np.zeros(self.n_classes)
        Sinv = [np.zeros([self.n_features,self.n_features]) for i in range(self.n_classes)] 
        Xm = [np.zeros(s2) for i in range(self.n_classes)] 
        XmTSinv = [np.zeros(s2) for i in range(self.n_classes)] 
        XmTSinvXm = np.zeros((self.n_classes))
        logPrior = np.zeros((self.n_classes))
        g_dscr = np.zeros((self.n_classes))
        
        for p in range(s1):
            x = X_test[p,:]
            for ic in range(self.n_classes):
                detS[ic] = np.linalg.det(self.covar_mat[ic])  
                logDetS[ic] = np.log(detS[ic] if detS[ic]>0 else 1e-6) 
                
                Sinv[ic] = np.linalg.pinv(self.covar_mat[ic])
                Xm[ic] = x - self.class_mean[ic]

                XmTSinv[ic] = np.matmul(Xm[ic],Sinv[ic])
                XmTSinvXm[ic] = np.matmul(XmTSinv[ic],np.transpose(Xm[ic])) 

                logPrior[ic] = np.log(self.prior_prob[ic]) 

                g_dscr[ic] = (-0.5*logDetS[ic]) + (-0.5*XmTSinvXm[ic]) + logPrior[ic] 
                
                y_pred[p] = np.argmax(g_dscr)
        
        return y_pred
        
        
    