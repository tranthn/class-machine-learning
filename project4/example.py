#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import math
import data_loader as dl

label = 'class'

def getProbsAndPreds(someX):
    probs = softmax(np.dot(someX,w))
    preds = np.argmax(probs,axis=1)
    return probs,preds

def oneHotIt(Y):
    return pd.get_dummies(Y, columns = ['class'])

def softmax(z):
    z -= np.max(z) # dimensions = n x k

    # denom, dimensions = n
    denom = np.sum(np.exp(z),axis=1)
    print('\ndenom')
    print(denom)
    sm = (np.exp(z).T / denom).T
    return sm

def getLoss(w, x, y, lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    print(w)
    print('\nx')
    print(x)
    print('\ny_mat')
    print(y_mat) # dimensions = n x k
    print('\nscores')
    print(scores) # dimensions = n x k
    print()
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    print('\nprob')
    print(prob) # dimensions = n x k
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad

def getAccuracy(someX, someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY) / (float(len(someY)))
    return accuracy

########### main softmax loop ###########
df = dl.get_iris_data()['folds'][0]
# df = dl.get_breast_data()['tune']

# dimensions = n x d
# should represent features only
x = df.drop(columns = ['class'])

# y = flat 1-d array of classes
y = df[label]

# dimensions = d x k
w = np.zeros([x.shape[1], len(np.unique(y))])
lam = 1
iterations = 1
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w, x, y, lam)
    losses.append(loss)
    w = w - (learningRate * grad)

print(losses)