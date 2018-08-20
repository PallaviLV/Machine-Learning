from __future__ import division, print_function
from typing import List, Tuple, Callable

import numpy as np
import scipy
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, nb_features=2, max_iteration=10, margin=1e-4):
        self.nb_features = nb_features
        self.w = [0 for i in range(0,nb_features+1)]
        self.margin = margin
        self.max_iteration = max_iteration

    def train(self, features: List[List[float]], labels: List[int]) -> bool:
        # stores the epsilon value
        eps_value = np.finfo(float).eps
  
        for i in range(self.max_iteration):
            j = -1
            flag = 1
            for x in features:
                j += 1
                
                #implementing Stochastic Gradient Descent
                x_norm = np.linalg.norm(x)
                w_norm = np.linalg.norm(self.w)
                w_arr = np.asarray(self.w)
                x_arr = np.asarray(x)
                value1 = (x_arr.transpose()).dot(w_arr) # stores xTw which is equal to wTx
                value2 = labels[j] * value1 # stores wTx * y
                value1 = value1 / (w_norm + eps_value) # stores wTx / (||w|| + epsilon)
                
                if((value1 >= (-self.margin/2) and value1 <= (self.margin/2)) or value2 < 0):
                    #should update w
                    flag = 0
                    self.w = self.w + ((labels[j] * x_arr) / x_norm)
                
            if(flag == 1):
                # all predicted labels for all feature vectors are correct i.e., there is convergence
                return True
        return False
    
    def reset(self):
        self.w = [0 for i in range(0,self.nb_features+1)]
        
    def predict(self, features: List[List[float]]) -> List[int]:
        pred_label = []
        for x in features:
            y_value = 0
            for j in range(0, len(x)):
                y_value += self.w[j] * x[j]
            if(y_value >= 0):
                pred_label.append(1)
            else:
                pred_label.append(-1)
                
        return pred_label
        
    def get_weights(self) -> List[float]:
        return self.w
    