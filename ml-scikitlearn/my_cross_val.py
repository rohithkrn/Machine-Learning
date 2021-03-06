from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

## Cross Validation

def my_cross_val(method,X,y,k):

    if method == "LinearSVC":
        myModel = LinearSVC()
        #print("MyModel is LinearSVC")
    elif method == "SVC":
        myModel = SVC()
        #print("MyModel is SVC")
    elif method == "LogisticRegression":
        myModel = LogisticRegression()
        #print("MyModel is LogisticRegression")
    else:
        print("Invalid Method")

    #myLinearSVC = LinearSVC()
    s1,s2 = X.shape
    accuracy = np.zeros(k)
    my_error = np.zeros(k)
    frac_size = (s1/k)
    
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
        #accuracy[i] = accuracy_score(y_test, y_pred)
       
    mean_error = np.mean(my_error)
    std_error = np.std(my_error)
    print("Error Rate:",my_error)
    print("Mean Error:",mean_error)
    print("Standard Deviation of Error:",std_error)
    return my_error
