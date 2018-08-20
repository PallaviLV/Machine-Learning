from typing import List

import numpy as np
from math import *

def mean_squared_error(y_true: List[float], y_pred: List[float]) -> float:
    assert len(y_true) == len(y_pred)
    
    arr_y_true = np.asarray(y_true)
    arr_y_pred = np.asarray(y_pred)
    
    squared_diff = []
    for i in range(0,len(arr_y_true)):
        squared_diff.append(pow((arr_y_pred[i] - arr_y_true[i]),2))
    
    return np.mean(squared_diff)


def f1_score(real_labels: List[int], predicted_labels: List[int]) -> float:
    assert len(real_labels) == len(predicted_labels)
    
    true_pos, false_pos, true_neg, false_neg = 0, 0, 0, 0
    # predicted_labels    real_labels
    #       1                 1        -> True Positive
    #       1                 0        -> False Positive
    #       0                 0        -> True Negative
    #       0                 1        -> False Negative
    
    for i in range(len(real_labels)):
        if(predicted_labels[i] == 1 and real_labels[i] == 1):
            true_pos += 1
        elif(predicted_labels[i] == 1 and real_labels[i] == 0):
            false_pos += 1
        elif(predicted_labels[i] == 0 and real_labels[i] == 0):
            true_neg += 1
        elif(predicted_labels[i] == 0 and real_labels[i] == 1):
            false_neg += 1
    #setting f1 to 0 when tp and fp both are 0, to avoid divide by zero error
    if(true_pos == false_pos == 0 or true_pos == false_neg == 0):
        return 0
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    f1_measure = 2 * (precision * recall) / (precision + recall)
    return f1_measure

    
def polynomial_features(
        features: List[List[float]], k: int
) -> List[List[float]]:

    # creating a new feature matrix X
    X = features
     
    # l keeps track of the power to which each feature column has to be raised 
    for l in range(2,k+1):
        for j in range(len(features[0])):
            new_feature = []
            for i in range(len(X)):
                poly = pow(X[i][j],l)
                new_feature.append(round(poly,6))               
            # the additonal feature columns created are being inserted into feature matrix X
            X = np.insert(X, len(features[0]) + j, new_feature, axis=1)   
    return X
            

def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    sum = 0  
    for i in range(len(point1)):
        sqrd_diff = pow((point1[i] - point2[i]),2)
        sum += sqrd_diff
    return sqrt(sum)


def inner_product_distance(point1: List[float], point2: List[float]) -> float:
    sum = 0
    for i in range(len(point1)):
        sum += point1[i]*point2[i]
    return sum


def gaussian_kernel_distance(
        point1: List[float], point2: List[float]
) -> float:
    sum = 0
    for i in range(len(point1)):
        sqrd_diff = pow((point1[i] - point2[i]),2)
        sum += sqrd_diff
    return -exp(-(1./2.)*sum)


class NormalizationScaler:
    def __init__(self):
        pass

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        norm_features = []
        for x in features:
            sum = 0
            norm_feature_vec = []
            # if all the features in x are 0, then all_zeroes variable would store True else False
            all_zeroes = all(ft == 0 for ft in x)
            # below if condition avoids feature vector with zeroes from being normalized
            if(not all_zeroes):
                for i in range(len(x)):
                    sum += x[i] * x[i]
                for i in range(len(x)):
                    norm_feature_vec.append(x[i] / sum)
            norm_features.append(norm_feature_vec)
        return norm_features

class MinMaxScaler:

    def __init__(self):
        self.count = 0
        self.min_max_values = []

    def __call__(self, features: List[List[float]]) -> List[List[float]]:
        # as per the examples given, min and max of each feature column in features matrix are obtained
        
        # using count to differentiate between training data (1st call) and test data
        self.count += 1
        
        if(self.count == 1):
            for i in range(len(features[0])):
                feature_list = []
                min_max_set = []
                for x in features:
                    feature_list.append(x[i])
                min_max_set.append(min(feature_list))
                min_max_set.append(max(feature_list))
                self.min_max_values.append(min_max_set)               
        # min_max_values contains the lists of min,max values for each column of features matrix
        
        #scaling the features
        for i in range(len(features[0])):
            diff = self.min_max_values[i][1] - self.min_max_values[i][0]
            for x in features:
                x[i] = (x[i] - self.min_max_values[i][0]) / diff
            
        return features
            