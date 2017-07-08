# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 12:55:36 2017

@author: rohith
"""
import numpy as np
from sklearn.linear_model import LogisticRegression

from my_cross_val import my_cross_val 

#%% Boston dataset
 
from sklearn.datasets import load_boston
Boston = load_boston()
y = np.sort(Boston.target)
size = np.size(Boston.target)

y50_val = np.array(y[round(size/2)])
y75_val = np.array(y[round(3*size/4)])

y50 = np.array(Boston.target > y50_val, dtype = int)
y75 = np.array(Boston.target > y75_val, dtype = int)

Boston50_data = np.array(Boston.data)
Boston50_target = np.array(y50)

Boston50_comb = np.c_[Boston50_data,Boston50_target]
Boston50_perm = np.random.permutation(Boston50_comb)

Boston50_data_perm = np.array(Boston50_perm[:,:-1])
Boston50_target_perm = np.array(Boston50_perm[:,-1])

Boston75_data = np.array(Boston.data)
Boston75_target = np.array(y75)

Boston75_comb = np.c_[Boston75_data,Boston75_target]
Boston75_perm = np.random.permutation(Boston75_comb)

Boston75_data_perm = np.array(Boston75_perm[:,:-1])
Boston75_target_perm = np.array(Boston75_perm[:,-1])

#%% Digits dataset

from sklearn.datasets import load_digits
digits = load_digits()

digits_data = digits.data
digits_target = digits.target


k = 5

#%% MultiGaussClassify - Boston50

print("MultiGaussClassify with Boston50")
err_MGC_b50 = my_cross_val("MultiGaussClassify",Boston50_data_perm,Boston50_target_perm,k)

print("MultiGaussClassify with Boston75")
err_MGC_b75 = my_cross_val("MultiGaussClassify",Boston75_data_perm,Boston75_target_perm,k)

print("MultiGaussClassify with Digits")
err_MGC_dig = my_cross_val("MultiGaussClassify",digits_data,digits_target,k)

print("LogisticRegression with Boston50")
err_LR_b50 = my_cross_val("LogisticRegression",Boston50_data_perm,Boston50_target_perm,k)

print("LogisticRegression with Boston75")
err_LR_b75 = my_cross_val("LogisticRegression",Boston75_data_perm,Boston75_target_perm,k)

print("LogisticRegression with Digits")
err_LR_dig = my_cross_val("LogisticRegression",digits_data,digits_target,k)




