from __future__ import division, print_function
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import cm

def binary_train(X, y, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    N, D = X.shape
    
    #Adding col of 0's to account for bias
    X = np.insert(X, 0, 1, 1)
    assert len(np.unique(y)) == 2
    b = 0
    if b0 is not None:
        b = b0

    #modified to account for bias 
    w = np.zeros(D+1)
    if w0 is not None:
        w = b
        w = np.append(w, w0)
           
    for j in range(max_iterations):
        total = np.zeros(D+1)
        for i in range(N):
            wTxn = np.dot(w.transpose(),X[i])
            error = sigmoid(wTxn) - y[i]
            total = total + np.dot(error.transpose(),X[i])

        # Using the average of the gradients for all training examples to update parameters.
        average = total/D
        # update rule
        w = w - step_size * average 
    b = w[0]
    w = w[1:]
    assert w.shape == (D,)
    return w, b


def binary_predict(X, w, b):
    N, D = X.shape
    preds = np.zeros(N) 

    for i in range(N):
        wTxn = np.dot(w,X[i]) + b
        sig = sigmoid(wTxn)
        if sig >= 0.5:
            preds[i] = 1
        else: 
            preds[i] = 0     
    assert preds.shape == (N,) 
    return preds


def multinomial_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    # Implementing a multinomial logistic regression for multiclass classification. 

    N, D = X.shape
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    w = np.insert(w, 0, 1, 1)
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0
    
    #Adding col of 0's to account for bias
    X = np.insert(X, 0, 1, 1)
    sftMax = np.zeros(C)
    grad = np.zeros((C,D+1))
    encoded_labels = np.zeros((C,N))
    
    # Doing 1-of-k encoding; C classes : 0 - C-1
    for i in range(C):
        for j in range(N):
            if y[j] == i:
                encoded_labels[i][j] = 1
            else:
                encoded_labels[i][j] = 0
        
    for i in range(max_iterations):
        #randomly choose a number between 0 and N-1, to choose xn randomly
        rand_value = np.random.randint(N)
        for j in range(C):                           
            #compute softmax for each class with respect to x[k]; softmax(y=j|xk)
            sftMax[j] = softmax(X[rand_value], w, b, j, C)
            error = sftMax[j] - encoded_labels[j][rand_value]
            grad[j] = error * X[rand_value]
              
            #updating the weights & bias 
            w[j] = w[j] - step_size * grad[j]
    
    b = w[:,0]
    w = w[:,1:D+1]
    
    assert w.shape == (C, D)
    assert b.shape == (C,)
    return w, b


def softmax(x, w, b, cls, C):
    #this function computes the softmax for given x vector and weights vector(including bias)
    # x vector, w = C x D, b vector, cls = [0, C-1], C = No. of classes
    
    wTxn = np.zeros(C)
    for i in range(C):
        wTxn[i] = np.dot(w[i],x) + b[i]
    
    wTxn = wTxn - np.amax(wTxn)
    mod_wTxn = np.exp(wTxn)
    denom = np.sum(mod_wTxn)
    numerator = np.exp(wTxn[cls])
    return numerator / denom

def multinomial_predict(X, w, b):
    # X : testing features
    N, D = X.shape
    C = w.shape[0]
    # preds is N dimensional vector of multiclass predictions
    preds = np.zeros(N) 
    pred_prob = np.zeros((C,N))
        
    for i in range(C):
        for j in range(N):
            pred_prob[i][j] = softmax(X[j], w, b, i, C)
           
    #pred_prod contains the softmax values; C x N matrix
    preds = np.argmax(pred_prob,axis=0)
    assert preds.shape == (N,)
    return preds


def OVR_train(X, y, C, w0=None, b0=None, step_size=0.5, max_iterations=1000):
    # Implementing multiclass classification using binary classifier and one-versus-rest strategy. 
    # OVR classifier is trained by training C different classifiers. 
    
    # X : Training features
    N, D = X.shape
    # w0: initial value of weight matrix
    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    # b0: initial value of bias term
    if b0 is not None:
        b = b0
    
    y_1kencoding = np.zeros(N)
    for i in range(C):
        # Doing 1-of-k encoding; C classes : 0 - C-1
        for j in range(N):
            if y[j] == i:
                y_1kencoding[j] = 1
            else:
                y_1kencoding[j] = 0
        w[i],b[i] = binary_train(X, y_1kencoding, w[i], b[i], step_size, max_iterations)
    assert w.shape == (C, D), 'wrong shape of weights matrix'
    assert b.shape == (C,), 'wrong shape of bias terms vector'
    return w, b


def OVR_predict(X, w, b):
    # X : testing features
    N, D = X.shape
    C = w.shape[0]
    # pred_prod contains the sigmoid values; C x N matrix
    # preds is a vector of class label predictions; {0, C-1}
    preds = np.zeros(N) 
    pred_prob = np.zeros((C,N))
            
    for i in range(C):
        for j in range(N):
            wTxn = np.dot(w[i],X[j]) + b[i]
            pred_prob[i][j] = sigmoid(wTxn)
    
    # Make predictions using OVR strategy and predictions from binary classifier. 
    preds = np.argmax(pred_prob,axis=0)
    assert preds.shape == (N,)
    return preds

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy_score(true, preds):
    return np.sum(true == preds).astype(float) / len(true)

def run_binary():
    from data_loader import toy_data_binary, data_loader_mnist 

    print('Performing binary classification on synthetic data')
    X_train, X_test, y_train, y_test = toy_data_binary()
    w, b = binary_train(X_train, y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' %(accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))
    
    print('Performing binary classification on binarized MNIST')
    X_train, X_test, y_train, y_test = data_loader_mnist()

    binarized_y_train = [0 if yi < 5 else 1 for yi in y_train] 
    binarized_y_test = [0 if yi < 5 else 1 for yi in y_test] 
    w, b = binary_train(X_train, binarized_y_train)
    
    train_preds = binary_predict(X_train, w, b)
    preds = binary_predict(X_test, w, b)
    print('train acc: %f, test acc: %f' %(accuracy_score(binarized_y_train, train_preds), accuracy_score(binarized_y_test, preds)))

def run_multiclass():
    from data_loader import toy_data_multiclass_3_classes_non_separable, toy_data_multiclass_5_classes, data_loader_mnist 
    
    datasets = [(toy_data_multiclass_3_classes_non_separable(), 'Synthetic data', 3), 
                (toy_data_multiclass_5_classes(), 'Synthetic data', 5), 
                (data_loader_mnist(), 'MNIST', 10)]

    for data, name, num_classes in datasets:
        print('%s: %d class classification' % (name, num_classes))
        X_train, X_test, y_train, y_test = data
        
        print('One-versus-rest:')
        w, b = OVR_train(X_train, y_train, C=num_classes)
        train_preds = OVR_predict(X_train, w=w, b=b)
        preds = OVR_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' %(accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))
    
        print('Multinomial:')
        w, b = multinomial_train(X_train, y_train, C=num_classes)
        train_preds = multinomial_predict(X_train, w=w, b=b)
        preds = multinomial_predict(X_test, w=w, b=b)
        print('train acc: %f, test acc: %f' %(accuracy_score(y_train, train_preds), accuracy_score(y_test, preds)))


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", )
    parser.add_argument("--output")
    args = parser.parse_args()

    if args.output:
            sys.stdout = open(args.output, 'w')

    if not args.type or args.type == 'binary':
        run_binary()

    if not args.type or args.type == 'multiclass':
        run_multiclass()