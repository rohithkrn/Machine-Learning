# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 14:35:46 2017

@author: rohit
"""
#%%
import numpy as np

#%%
class MyFLDA2():
    
    # Data features
    classes = 0
    count = 0 
    n_classes = 0 
    n_samples = 0
    n_features = 0

    # Model parameters
    class_mean = 0
    covar_mat = 0
    W = 0
    error = 0
    thresh = 0
    def __init__(self):

        #self.class_mean = np.zeros(self.n_features)
        self.W = np.zeros(self.n_features)
        self.thresh = 0
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
        
        #self.prior_prob = np.zeros(self.n_classes)
        self.class_mean = [np.zeros(self.n_features) for i in range(self.n_classes)] 
        class_var = np.zeros([self.n_features,self.n_features])
        self.covar_mat = [np.zeros([self.n_features,self.n_features]) for i in range(self.n_classes)]
        SW = np.zeros([self.n_features,self.n_features])

        for ii in range(self.n_classes):
            #self.prior_prob[ii] = self.count[ii]/self.n_samples
            self.class_mean[ii] = np.mean(strat_classes[ii],axis = 0) 
            mean_cent = strat_classes[ii] - self.class_mean[ii]
            class_var = np.matmul(np.transpose(mean_cent),mean_cent)
            class_var = class_var/len(strat_classes[ii])
            self.covar_mat[ii] = class_var
            SW = SW + self.covar_mat[ii]

         
        self.W = np.matmul(np.linalg.inv(SW),(self.class_mean[0] - self.class_mean[1]))
        z1_tr = np.matmul(strat_classes[0],self.W)
        z2_tr = np.matmul(strat_classes[1],self.W)
        
        s1 = z1_tr.shape[0]
        s2 = z2_tr.shape[0]
        z1_mean = np.mean(z1_tr)
        z2_mean = np.mean(z2_tr)

        self.thresh = (z1_mean*s2 + z2_mean*s1)/self.n_samples
    ## Prdiction    
        
    def predict(self,X_test):

        s1,s2 = X_test.shape
        y_pred = np.zeros(s1)

             
        for p in range(s1):
            x = X_test[p,:]
            x_tr = np.matmul(x,self.W)
            if x_tr < self.thresh:
               y_pred[p] = 1 
            #    y_pred[p] = np.argmax(g_dscr)
        
        return y_pred
        
        
    