from sklearn.datasets import fetch_openml
import pandas as pd
from multiprocessing import Pool, cpu_count
import time
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sys

def KNN_clf(n_neighbors, algorithm, p, X_train, y_train, X_test, y_test):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, algorithm = algorithm, p = p)
    clf.fit(X_train, y_train)
    error_rate = (1 - clf.score(X_test , y_test))
    st = 'Algorithm: '+algorithm+" , n_neighbors: "+ str(n_neighbors) +" , p: " + str(p)
    st = st + '======> error rate ' + str(error_rate)
    print(st)
    with open('knn_output_true.txt', 'a') as f:
        f.writelines(st+ '\n')



if __name__ == '__main__':
    n_neighbors = int(sys.argv[1])
    algorithm = sys.argv[2]
    p = int(sys.argv[3])
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    KNN_clf(n_neighbors, algorithm, p, X_train, y_train, X_test, y_test)
    print("Hurray")
