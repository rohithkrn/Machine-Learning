import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pandas import DataFrame

from rand_proj import rand_proj
from quad_proj import quad_proj
from my_cross_val import my_cross_val

from sklearn.datasets import load_digits
digits = load_digits()

digits_data = digits.data
digits_target = digits.target

X1 = rand_proj(digits_data,32)
X2 = quad_proj(digits_data)
X3 = rand_proj(X2,64)

#LinearSVC
print("LinearSVC Cross-Validation with X1")
err_X1_LSVC_CV = my_cross_val("LinearSVC",X1,digits_target,10)
mean_err_X1_LSVC_CV = np.mean(err_X1_LSVC_CV)
std_err_X1_LSVC_CV = np.std(err_X1_LSVC_CV)

print("LinearSVC Cross-Validation with X2")
err_X2_LSVC_CV = my_cross_val("LinearSVC",X2,digits_target,10)
mean_err_X2_LSVC_CV = np.mean(err_X2_LSVC_CV)
std_err_X2_LSVC_CV = np.std(err_X2_LSVC_CV)

print("LinearSVC Cross-Validation with X3")
err_X3_LSVC_CV = my_cross_val("LinearSVC",X3,digits_target,10)
mean_err_X3_LSVC_CV = np.mean(err_X3_LSVC_CV)
std_err_X3_LSVC_CV = np.std(err_X3_LSVC_CV)

#SVC
print("SVC Cross-Validation with X1")
err_X1_SVC_CV = my_cross_val("SVC",X1,digits_target,10)
mean_err_X1_SVC_CV = np.mean(err_X1_SVC_CV)
std_err_X1_SVC_CV = np.std(err_X1_SVC_CV)

print("SVC Cross-Validation with X2")
err_X2_SVC_CV = my_cross_val("SVC",X2,digits_target,10)
mean_err_X2_SVC_CV = np.mean(err_X2_SVC_CV)
std_err_X2_SVC_CV = np.std(err_X2_SVC_CV)

print("SVC Cross-Validation with X3")
err_X3_SVC_CV = my_cross_val("SVC",X3,digits_target,10)
mean_err_X3_SVC_CV = np.mean(err_X3_SVC_CV)
std_err_X3_SVC_CV = np.std(err_X3_SVC_CV)

#LogisticRegression
print("LogisticRegression Cross-Validation with X1")
err_X1_LR_CV = my_cross_val("LogisticRegression",X1,digits_target,10)
mean_err_X1_LR_CV = np.mean(err_X1_LR_CV)
std_err_X1_LR_CV = np.std(err_X1_LR_CV)

print("LogisticRegression Cross-Validation with X2")
err_X2_LR_CV = my_cross_val("LogisticRegression",X2,digits_target,10)
mean_err_X2_LR_CV = np.mean(err_X2_LR_CV)
std_err_X2_LR_CV = np.std(err_X2_LR_CV)

print("LogisticRegression Cross-Validation with X3")
err_X3_LR_CV = my_cross_val("LogisticRegression",X3,digits_target,10)
mean_err_X3_LR_CV = np.mean(err_X3_LR_CV)
std_err_X3_LR_CV = np.std(err_X3_LR_CV)


## Export Mean to Excel file
df1 = DataFrame({'X1_LSVC': err_X1_LSVC_CV, 'X2_LSVC': err_X2_LSVC_CV,'X3_LSVC': err_X3_LSVC_CV})
df2 = DataFrame({'X1_SVC': err_X1_SVC_CV, 'X2_SVC': err_X2_SVC_CV,'X3_SVC': err_X3_SVC_CV})
df3 = DataFrame({'X1_LR': err_X1_LR_CV, 'X2_LR': err_X2_LR_CV,'X3_LR': err_X3_LR_CV})

df1.to_excel('q4_LinearSVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df2.to_excel('q4_SVC_Errors.xlsx', sheet_name ='sheet1', index=False)
df3.to_excel('q4_LogRegression_Errors.xlsx', sheet_name ='sheet1', index=False)
