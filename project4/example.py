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
    # m = Y.shape[0]
    # OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
    # OHX = np.array(OHX.todense()).T
    # return OHX
    return pd.get_dummies(Y, columns = ['class'])

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm

def getLoss(w, x, y, lam):
    m = x.shape[0] #First we get the number of training examples
    y_mat = oneHotIt(y) #Next we convert the integer class coding into a one-hot representation
    scores = np.dot(x,w) #Then we compute raw class scores given our input and current weights
    prob = softmax(scores) #Next we perform a softmax on these scores to get their probabilities
    loss = (-1 / m) * np.sum(y_mat * np.log(prob)) + (lam/2)*np.sum(w*w) #We then find the loss of the probabilities
    grad = (-1 / m) * np.dot(x.T,(y_mat - prob)) + lam*w #And compute the gradient for that loss
    return loss,grad

def getAccuracy(someX,someY):
    prob,prede = getProbsAndPreds(someX)
    accuracy = sum(prede == someY)/(float(len(someY)))
    return accuracy

########### main softmax loop ###########
df = dl.get_iris_data()['tune']

# should represent features only
x = df.drop(columns = ['class'])

# y = flat 1-d array of classes
y = df[label]

w = np.zeros([x.shape[1], len(np.unique(y))])
lam = 1
iterations = 1000
learningRate = 1e-5
losses = []
for i in range(0,iterations):
    loss,grad = getLoss(w, x, y, lam)
    losses.append(loss)
    w = w - (learningRate * grad)

print(loss)
