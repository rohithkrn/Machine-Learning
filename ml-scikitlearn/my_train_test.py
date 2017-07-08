from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

def my_train_test(method,X,y,pi,k):

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
        
    s1,s2 = X.shape
    Xy_comb = np.c_[X,y]
    accuracy = np.zeros(k)
    my_error = np.zeros(k)

    for i in range(k):
        error = 0
        Xy_perm = np.random.permutation(Xy_comb)

        X_perm = np.array(Xy_perm[:,:-1])
        y_perm = np.array(Xy_perm[:,-1])

        X_train = np.array(X_perm[:pi*s1,:])
        X_test = np.array(X_perm[pi*s1:,:])

        y_train = np.array(y_perm[:pi*s1])
        y_test = np.array(y_perm[pi*s1:])

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