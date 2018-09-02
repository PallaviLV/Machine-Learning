## Programming Assignment 2

In this assignment I have implemented Binary and Multiclass classification and Neural Networks.

### Logistic Regression
In **logistic.py**, I have implemented Logistic Regression for Binary and Multiclass classification.
Note: Since Logistic Regression does not have a closed form solution, used Gradient Descent for training.

#### Binary Classification
To execute this, run the scripts **logistic_binary.sh** and **logistic_multiclass.sh**, which will output **logistic_binary.out** and **logistic_multiclass.out**  

#### Multiclass Classification
Two approaches:

(a) One-Versus-Rest classification
    For K classes we train K classifiers. Each classifier is trained on a binary problem, where its either belonging to one class 
    or belonging to any other class. The multiclass prediction is made based on the combination of all predictions from K binary
    classiers.
    *OVR_train* and *OVR_predict* functions perform One-Verus-Rest classification.

(b) Multinomial Logistic Regression
    Note: Use softmax function for conditional probability
    Train K binary classifiers and each point is assigned to a class, that maximizes the conditional probability.
    *multinomial_train* and *multinomial_predict* are the functions used for this.

Run **logistic_multiclass.sh** script, which will produce **logistic_multiclass.out**. 

### Neural Networks
Modules are elements of a Multi-Layer Perceptron (MLP) or Convolutional Neural Network (CNN). Each module is defined as a Class. 
Each module can perform a forward pass and a backward pass. The forward pass performs the computation of the module,
given the input to the module. The backward pass computes the partial derivatives of the loss function w.r.t. the input and parameters, 
given the partial derivatives of the loss function w.r.t. the output of the module.

Here we train the models using stochastic gradient descent with minibatch, and explore how different hyperparameters 
of optimizers and regularization techniques affect training and validation accuracies over training epochs. 
For deeper understanding, check out, e.g., the seminal work of Yann LeCun et al. "Gradient-based learning applied to document recognition" written in 1998.

#### Testing MLP
* Run script **q33.sh**. It will output **MLP_lr0.01_m0.0_w0.0_d0.0.json**.
    **q33.sh** will run **python3 dnn_mlp.py** with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.0. 
    The output file stores the training and validation accuracies over 30 training epochs.
* Run script **q34.sh**. It will output **MLP_lr0.01_m0.0_w0.0_d0.5.json**.
    **q34.sh** will run **python3 dnn_mlp.py --dropout rate 0.5** with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.
* Run script **q35.sh**. It will output **MLP_lr0.01_m0.0_w0.0_d0.95.json**.
    **q35.sh** will run **python3 dnn_mlp.py --dropout rate 0.95** with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.95. The output file stores the training and validation accuracies over 30 training epochs.

We will observe that the model in q34 will give better validation accuracy (at epoch 30) compared to q33. Specically, 
dropout is widely-used to prevent over-fitting. However, if we use a too large dropout rate (like the one in q35), 
the validation accuracy (together with the training accuracy) will be relatively lower, essentially under-fitting the training data.

* Run script **q36.sh**. It will output **LR_lr0.01_m0.0_w0.0_d0.0.json**.
    **q36.sh** will run **python3 dnn_mlp_nononlinear.py** with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.0. 
    The output file stores the training and validation accuracies over 30 training epochs.

The network has the same structure as the one in q33, except that we remove the relu (nonlinear) layer. We will see that the 
validation accuracies drop signicantly (the gap is around 0.03). Essentially, without the nonlinear layer, the model is learning 
multinomial logistic regression.

#### Testing CNN
* Run script **q37.sh**. It will output **CNN_lr0.01_m0.0_w0.0_d0.5.json**.
    **q37.sh** will run **python3 dnn_cnn.py** with learning rate 0.01, no momentum, no weight decay, and dropout rate 0.5. 
    The output file stores the training and validation accuracies over 30 training epochs.
* Run script **q38.sh**. It will output **CNN_lr0.01_m0.9_w0.0_d0.5.json**.
    **q38.sh** will run **python3 dnn_cnn.py --alpha 0.9** with learning rate 0.01, momentum 0.9, no weight decay, and dropout rate 0.5. 
    The output file stores the training and validation accuracies over 30 training epochs.

We will see that q38 will lead to faster convergence than q37 (i.e., the training/validation accuracies will be higher than 0.94 after 1 epoch). 
That is, using momentum will lead to more stable updates of the parameters.

#### Deeper architecture
**dnn_cnn.py** has only one convolutional layer. **dnn_cnn2.py** will construct a two convolutional layer CNN.

* Run script **q310.sh**. It will output **CNN2_lr0.001_m0.9_w0.0_d0.5.json**.
    **q310.sh** will run **python3 dnn_cnn_2.py --alpha 0.9** with learning rate 0.01, momentum 0.9, no weight decay, and dropout rate 0.5. The output file stores the training and validation accuracies over 30 training epochs.

We see that this achieves slightly higher validation accuracies than in q38.

The forward and backward functions of each layer/module are implemented in **dnn_misc.py**. Files **dnn_mlp.py** and **dnn_cnn.py** construct the MLP and CNN.

