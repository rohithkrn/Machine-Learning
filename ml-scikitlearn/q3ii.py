import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame


#from my_cross_val import my_cross_val
from my_train_test import my_train_test

from sklearn.datasets import load_boston
Boston = load_boston()

y = np.sort(Boston.target)
size = np.size(Boston.target)

y50_val = np.array(y[size/2])
y75_val = np.array(y[3*size/4])

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

from sklearn.datasets import load_digits
digits = load_digits()

digits_data = digits.data
digits_target = digits.target


k=10
pi=0.75
#LinearSVC
print("LinearSVC Train Test Split with Boston50")
err_b50_LSVC_TT = my_train_test("LinearSVC",Boston50_data_perm,Boston50_target_perm,pi,k)
mean_err_b50_LSVC_TT = np.mean(err_b50_LSVC_TT)
std_err_b50_LSVC_TT = np.std(err_b50_LSVC_TT)

print("LinearSVC Train Test Split with Boston75")
err_b75_LSVC_TT = my_train_test("LinearSVC",Boston75_data_perm,Boston75_target_perm,pi,k)
mean_err_b75_LSVC_TT = np.mean(err_b75_LSVC_TT)
std_err_b75_LSVC_TT = np.std(err_b75_LSVC_TT)

print("LinearSVC Train Test Split with Digits")
err_digit_LSVC_TT = my_train_test("LinearSVC",digits_data,digits_target,pi,k)
mean_err_digit_LSVC_TT = np.mean(err_digit_LSVC_TT)
std_err_digit_LSVC_TT = np.std(err_digit_LSVC_TT)

#SVC
print("SVC Train Test Split with Boston50")
err_b50_SVC_TT = my_train_test("SVC",Boston50_data_perm,Boston50_target_perm,pi,k)
mean_err_b50_SVC_CV = np.mean(err_b50_SVC_TT)
std_err_b50_SVC_CV = np.std(err_b50_SVC_TT)

print("SVC Train Test Split with Boston75")
err_b75_SVC_TT = my_train_test("SVC",Boston75_data_perm,Boston75_target_perm,pi,k)
mean_err_b75_SVC_TT = np.mean(err_b75_SVC_TT)
std_err_b75_SVC_TT = np.std(err_b75_SVC_TT)


print("SVC Train Test Split with Digits")
err_digit_SVC_TT = my_train_test("SVC",digits_data,digits_target,pi,k)
mean_err_digit_SVC_TT = np.mean(err_digit_SVC_TT)
std_err_digit_SVC_TT = np.std(err_digit_SVC_TT)


#Logistic Regression
print("LogisticRegression Train Test Split with Boston50")
err_b50_LR_TT = my_train_test("LogisticRegression",Boston50_data_perm,Boston50_target_perm,pi,k)
mean_err_b50_LR_TT = np.mean(err_b50_LR_TT)
std_err_b50_LR_TT = np.std(err_b50_LR_TT)

print("LogisticRegression Train Test Split with Boston75")
err_b75_LR_TT = my_train_test("LogisticRegression",Boston75_data_perm,Boston75_target_perm,pi,k)
mean_err_b75_LR_TT = np.mean(err_b75_LR_TT)
std_err_b75_LR_TT = np.std(err_b75_LR_TT)

print("LogisticRegression Train Test Split with Digits")
err_digit_LR_TT = my_train_test("LogisticRegression",digits_data,digits_target,pi,k)
mean_err_digit_LR_TT = np.mean(err_digit_LR_TT)
std_err_digit_LR_TT = np.std(err_digit_LR_TT)

## Export Errors to Excel File

df1 = DataFrame({'b50_LSVC': err_b50_LSVC_TT, 'b75_LSVC': err_b75_LSVC_TT,'digit_LSVC': err_digit_LSVC_TT})
df2 = DataFrame({'b50_SVC': err_b50_SVC_TT, 'b75_SVC': err_b75_SVC_TT,'digit_SVC': err_digit_SVC_TT})
df3 = DataFrame({'b50_LR': err_b50_LR_TT, 'b75_LR': err_b75_LR_TT,'digit_LR': err_digit_LR_TT})

df1.to_excel('q3ii_LinearSVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df2.to_excel('q3ii_SVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df3.to_excel('q3ii_LogRegression_Erros.xlsx', sheet_name ='sheet1', index=False)