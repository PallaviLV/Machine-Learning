from __future__ import division, print_function
from typing import List, Callable

import numpy
import scipy

class KNN:
    def __init__(self, k: int, distance_function) -> float:
        self.k = k
        self.distance_function = distance_function

    def train(self, features: List[List[float]], labels: List[int]):
        # storing the training features and corresponding labels
        self.features = features
        self.labels = labels

    def predict(self, features: List[List[float]]) -> List[int]:
        predicted_labels = []
  
        for x in features:
            distances = []
            all_indices = []
            
            for i in range(len(self.features)):
                # leaving out the current feature vector
                if( x != self.features[i]):
                    dis = self.distance_function(x, self.features[i])
                    distances.append(dis)
                    all_indices.append(i)
            
            indices = []
            actual_indices = []
            dis_arr = numpy.asarray(distances)
            # to find k nearest neighbours
            indices = dis_arr.argsort()[:self.k]
            
            # getting the actual indices of k nearest neighbours in features matrix
            for j in indices:
                actual_indices.append(all_indices[j])
            # to classify the features into 2 classes {0,1}
            C0, C1 = 0, 0
            for j in actual_indices:
                if(self.labels[j] == 1):
                    C1 += 1
                else:
                    C0 += 1
            # assigning this feature to the class with highest number of votes
            if(C1 > C0):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
       
        return predicted_labels
    
if __name__ == '__main__':
    print(numpy.__version__)
    print(scipy.__version__)
