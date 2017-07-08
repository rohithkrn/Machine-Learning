from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
import numpy as np

from MultiGaussClassify import MultiGaussClassify

## Cross Validation

def my_cross_val(method,X,y,k):

    if method == "MultiGaussClassify":
        myModel = MultiGaussClassify()
        #print("MyModel is MultiGaussClassify")
    elif method == "LogisticRegression":
        myModel = LogisticRegression()
        #print("MyModel is LogisticRegression")
    else:
        print("Invalid Method")

    s1,s2 = X.shape
    #accuracy = np.zeros(k)
    my_error = np.zeros(k)
    frac_size = round(s1/k)
    
    for i in range(k):
        error = 0
        X_test = X[i*frac_size:(i+1)*frac_size,:]
        X_train = np.concatenate((X[:i*frac_size,:],X[(i+1)*frac_size:,:]),axis = 0)

        y_test = y[i*frac_size:(i+1)*frac_size]       
        y_train = np.concatenate((y[:i*frac_size],y[(i+1)*frac_size:]),axis = 0)
        

        myModel.fit(X_train,y_train)
       
        y_pred = myModel.predict(X_test)

        for ii in range(y_pred.size):
            if y_pred[ii] != y_test[ii]:
                error += 1
        
        my_error[i] = (error)/y_pred.size
        print("Error for fold",i+1,":",my_error[i])
       
    mean_error = np.mean(my_error)
    std_error = np.std(my_error)
    print("Mean Error:",mean_error)
    print("Standard Deviation of Error:",std_error)
    return my_error
