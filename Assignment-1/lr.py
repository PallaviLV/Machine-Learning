from __future__ import division, print_function
from typing import List

import numpy
import scipy

class LinearRegression:
    def __init__(self, nb_features: int):
        self.nb_features = nb_features
        self.w = []

    def train(self, features: List[List[float]], values: List[float]):
        # inserting a column of 1s at index=0, so the bias term is also considered when computing wTX (w0*1 = w0)
        X = numpy.insert(features, 0, 1, axis=1)          
        sqr_mat = (X.transpose()).dot(X)
        sqr_mat_inv = numpy.linalg.inv(sqr_mat)
        #this gives me the LMS weights
        self.w = (sqr_mat_inv.dot(X.transpose())).dot(values)

    def predict(self, features: List[List[float]]) -> List[float]:
        X = numpy.insert(features, 0, 1, axis=1)
        y_pred = []

        for x in X:
            y_value = 0
            for j in range(0, len(x)):
                y_value += self.w[j] * x[j]
            y_pred.append(y_value)
        return y_pred
            
    def get_weights(self) -> List[float]:
        return self.w


class LinearRegressionWithL2Loss:
    '''Use L2 loss for weight regularization'''
    def __init__(self, nb_features: int, alpha: float):
        self.alpha = alpha
        self.nb_features = nb_features
        self.w=[]

    def train(self, features: List[List[float]], values: List[float]):
        X = numpy.insert(features, 0, 1, axis=1)
        sqr_mat = (X.transpose()).dot(X)
        # getting an Identity matrix of size = No. of features
        identy_mat = numpy.identity(len(X[0]))
        matrix_sum = sqr_mat + (self.alpha * identy_mat) 
        sqr_mat_inv = numpy.linalg.inv(matrix_sum)
        #this gives me the RLS weights
        self.w = (sqr_mat_inv.dot(X.transpose())).dot(values)

    def predict(self, features: List[List[float]]) -> List[float]:
        X = numpy.insert(features, 0, 1, axis=1)
        y_pred = []
        for x in X:
            y_value = 0
            for j in range(0, len(self.w)):
                y_value += self.w[j] * x[j]
            y_pred.append(y_value)
        return y_pred

    def get_weights(self) -> List[float]:
        return self.w

if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
