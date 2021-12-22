from sklearn.datasets import fetch_openml
import pandas as pd
from multiprocessing import Pool, cpu_count
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import sys


def NN_clf(solver, activation, hidden_layer_sizes, max_iter, alpha, X_train, y_train, X_test, y_test):
    clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, solver=solver, activation=activation, max_iter=max_iter, alpha = alpha)
    clf.fit(X_train, y_train)
    error_rate = (1 - clf.score(X_test , y_test))
    st = 'solver: '+solver+" , activation: "+ activation +" , hidden_layer_size: " + str(hidden_layer_sizes) + " , max_iter: " + str(max_iter) + " , alpha: " + str(alpha)
    st = st + ', error rate: ' + str(error_rate)
    print(st)
    with open('nn_output.txt', 'a') as f:
        f.writelines(st+ '\n')



if __name__ == '__main__':
    report_mode = int(sys.argv[1])
    solver = sys.argv[2]
    activation = sys.argv[3]
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X / 255
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    if report_mode == 1:
        for hidden_layer_sizes in [(128,64),(512,128,64), (1024,512,128,64)]:
            for max_iter in [50, 100, 200, 500]:
                for alpha in [0.001, 0.01, 0.1]:
                    NN_clf(solver, activation, hidden_layer_sizes, max_iter, alpha, X_train, y_train, X_test, y_test)
    else:
        max_iter = int(sys.argv[4])
        alpha = float(sys.argv[5])
        for hidden_layer_sizes in [(128,64)]:
            NN_clf(solver, activation, hidden_layer_sizes, max_iter, alpha, X_train, y_train, X_test, y_test)
    print("Completed")
