import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame


from my_cross_val import my_cross_val
#from my_functions import my_train_test

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

#LinearSVC
print("LinearSVC Cross-Validation with Boston50")
err_b50_LSVC_CV = my_cross_val("LinearSVC",Boston50_data_perm,Boston50_target_perm,k)
mean_err_b50_LSVC_CV = np.mean(err_b50_LSVC_CV)
std_err_b50_LSVC_CV = np.std(err_b50_LSVC_CV)

print("LinearSVC Cross-Validation with Boston75")
err_b75_LSVC_CV = my_cross_val("LinearSVC",Boston75_data_perm,Boston75_target_perm,k)
mean_err_b75_LSVC_CV = np.mean(err_b75_LSVC_CV)
std_err_b75_LSVC_CV = np.std(err_b75_LSVC_CV)

print("LinearSVC Cross-Validation with Digits")
err_digit_LSVC_CV = my_cross_val("LinearSVC",digits_data,digits_target,k)
mean_err_digit_LSVC_CV = np.mean(err_digit_LSVC_CV)
std_err_digit_LSVC_CV = np.std(err_digit_LSVC_CV)

#SVC
print("SVC Cross-Validation with Boston50")
err_b50_SVC_CV = my_cross_val("SVC",Boston50_data_perm,Boston50_target_perm,k)
mean_err_b50_SVC_CV = np.mean(err_b50_SVC_CV)
std_err_b50_SVC_CV = np.std(err_b50_SVC_CV)

print("SVC Cross-Validation with Boston75")
err_b75_SVC_CV = my_cross_val("SVC",Boston75_data_perm,Boston75_target_perm,k)
mean_err_b75_SVC_CV = np.mean(err_b75_SVC_CV)
std_err_b75_SVC_CV = np.std(err_b75_SVC_CV)


print("SVC Cross-Validation with Digits")
err_digit_SVC_CV = my_cross_val("SVC",digits_data,digits_target,k)
mean_err_digit_SVC_CV = np.mean(err_digit_SVC_CV)
std_err_digit_SVC_CV = np.std(err_digit_SVC_CV)


#Logistic Regression
print("LogisticRegression Cross-Validation with Boston50")
err_b50_LR_CV = my_cross_val("LogisticRegression",Boston50_data_perm,Boston50_target_perm,k)
mean_err_b50_LR_CV = np.mean(err_b50_LR_CV)
std_err_b50_LR_CV = np.std(err_b50_LR_CV)

print("LogisticRegression Cross-Validation with Boston75")
err_b75_LR_CV = my_cross_val("LogisticRegression",Boston75_data_perm,Boston75_target_perm,k)
mean_err_b75_LR_CV = np.mean(err_b75_LR_CV)
std_err_b75_LR_CV = np.std(err_b75_LR_CV)

print("LogisticRegression Cross-Validation with Digits")
err_digit_LR_CV = my_cross_val("LogisticRegression",digits_data,digits_target,k)
mean_err_digit_LR_CV = np.mean(err_digit_LR_CV)
std_err_digit_LR_CV = np.std(err_digit_LR_CV)

## Export Errors to Excel File

df1 = DataFrame({'b50_LSVC': err_b50_LSVC_CV, 'b75_LSVC': err_b75_LSVC_CV,'digit_LSVC': err_digit_LSVC_CV})
df2 = DataFrame({'b50_SVC': err_b50_SVC_CV, 'b75_SVC': err_b75_SVC_CV,'digit_SVC': err_digit_SVC_CV})
df3 = DataFrame({'b50_LR': err_b50_LR_CV, 'b75_LR': err_b75_LR_CV,'digit_LR': err_digit_LR_CV})

df1.to_excel('q3i_LinearSVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df2.to_excel('q3i_SVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df3.to_excel('q3i_LogRegression_Erros.xlsx', sheet_name ='sheet1', index=False)

