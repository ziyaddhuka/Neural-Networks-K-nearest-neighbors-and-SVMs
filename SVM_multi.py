from sklearn.datasets import fetch_openml
import pandas as pd
import sys
from sklearn.svm import SVC

def SVM_clf(C, gamma, kernel, X_train, y_train, X_test, y_test):
    clf = SVC(C = C, gamma = gamma, kernel=kernel, cache_size=1000)
    clf.fit(X_train, y_train)
    error_rate = (1 - clf.score(X_test , y_test))
    st = 'C: '+str(C)+" , gamma: "+ str(gamma) +" , kernel: " + kernel
    st = st + ' , error rate ' + str(error_rate)
    print(st)
    with open('SVC_output_true.txt', 'a') as f:
        f.writelines(st+ '\n')


if __name__ == '__main__':
    C = int(sys.argv[1])
    if sys.argv[2].replace(".", "", 1).isdigit():
        gamma = float(sys.argv[2])
    else:
        gamma = sys.argv[2]
    kernel = str(sys.argv[3])
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    SVM_clf(C, gamma, kernel, X_train, y_train, X_test, y_test)
    print("Completed")
