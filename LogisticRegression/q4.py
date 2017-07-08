import numpy as np
from sklearn.linear_model import LogisticRegression
from my_cross_val import my_cross_val

np.seterr(divide='ignore', invalid='ignore', over='ignore')

#%% Digits dataset

from sklearn.datasets import load_digits
digits = load_digits()

digits_data = digits.data
digits_target = digits.target

#%% MyLogisticRegGen - Boston50

k=5
print("MyLogisticRegGen with Digits")
err_MyLR_digits = my_cross_val("MyLogisticRegGen",digits_data ,digits_target,k)

print("LogisticRegression with Digits")
err_LR_b50 = my_cross_val("LogisticRegression",digits_data ,digits_target,k)
