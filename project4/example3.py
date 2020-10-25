#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math
import data_loader as dl

# source: https://nthu-datalab.github.io/ml/labs/04-1_Perceptron_Adaline/04-1_Perceptron_Adaline.html

class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the training dataset.
    random_state : int
        The seed of the pseudo random number generator.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    errors_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : array-like; shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like; shape = [n_samples]
            Target values or labels.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.activation(X)
            
            # Cost function
            error = (y - output)
            cost = (error**2).sum() / 2.0
            self.cost_.append(cost)
            
            print('\ny shape')
            print(y.shape)
            print(y)
            print('\noutput shape')
            print(output.shape)
            print(output)
            print('\nw[1:] shape')
            print(self.w_[1:].shape)
            
            print('\nX.t shape')
            print(X.T.shape)
            print(X.T)
            print()
            print('\nerror shape')
            print(error.shape)
            print(error)

            # Update rule
            print('\neta')
            print(self.eta)
            print('\ninner')
            print(self.eta * X.T.dot(error))
            print()

            self.w_[1:] += self.eta * X.T.dot(error)

            print('\nupdated w shape')
            print(self.w_[1:].shape)

            self.w_[0] += self.eta * error.sum()

            print('\nw[0]')
            print(self.w_[0])
            
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return self.net_input(X)

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

### main
# data = dl.get_iris_data()
data = dl.get_breast_data()
train = data['folds'][0]
test = data['folds'][1]
x = train.drop(columns = ['class'])
x_test = test.drop(columns = ['class'])
y_test = test['class']
y,levels = train['class'].factorize()

adas = AdalineGD(n_iter=1, eta=0.01)
model = adas.fit(x.to_numpy(), y)
out = model.predict(x_test)
# print('\ny_test')
# print(y_test)
# print('\nout')
# print(out)
