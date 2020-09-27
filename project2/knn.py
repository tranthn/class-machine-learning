#!/usr/bin/env python3
import sys
import math
import numpy as np
import pandas as pd

### for a given point p1, look at # k nearest points to p1
### distance/class of near neighbors and majority class
### we will label the point p1 with said class

### picking k
# k = odd, for 2-class problem
# k = not multiple of # classes
#################################

# calculates euclidean distance given 2 pandas dataframe rows
#
# arguments
#   - v1: row vector representing an instance
#   - v2: row vector representing an instance
#
# returns
#   - unrounded euclidean distance
def euclidean_dist(v1, v2, squared = False):
    square = lambda x : x ** 2

    # get array representing pairwise subtraction of vector elements
    v = np.subtract(v1, v2)

    # apply our lambda fn to square all values
    arr = np.array([square(vi) for vi in v])
    dist = np.sum(arr)

    if not squared:
        dist = math.sqrt(np.sum(arr))

    return dist

# calculate radial basis function kernel
#
# K(x, x') = exp( - [ |x - x'|^2] /[2sigma^2] ) 
#
# arguments
#   - v1: row vector representing an instance
#   - v2: row vector representing an instance
#
# returns
#   - XYZ
def rbf(v1, v2, sigma):
    dist = euclidean_dist(v1, v2, squared = True)
    gamma = 1 / (2 * (sigma ** sigma))
    k = -dist * gamma
    k2 = np.exp(k)
    return k2

#  modifies dataframe to add calculated distance column w.r.t target point
#
# arguments
#   - training: data frame representing our training rows
#   - point: target instance we're going to predict class for, via euclidean distance
#
# returns
#   - modified dataframe
def build_distance_table(train, point):
    train_dist = train.copy()
    train_dist['dist'] = train.apply(lambda x : euclidean_dist(point, x), axis = 1)
    return train_dist

#  determines which class is the majority among the picked neighbors
#  allows ties and just picks first option
#
# arguments
#   - neighbors: k-neighbors, # passed in determined by caller
#   - label: class or predicted column name
#
# returns
#   - predicted class
def majority_vote(neighbors, label):
    # value_counts() will return in descending count by default
    nn = neighbors[label].value_counts()
    best_neighbor = nn.index[0]

    return best_neighbor

# helper that builds table and finds neighbors for given training set x query point
#
# arguments
#   - training: dataframe of training data (4 folds for our 5-fold case)
#   - query: holdout fold or tuning set we're going to predict
#   - label: class or predicted column name
#   - k: number of neighbors we will select
#
# returns
#   - k nearest neighbors
def _find_knn(train, query, label, k):
    table = build_distance_table(train.drop(columns = label), query.drop(labels = label))

    # merge back to ensure the table has both the class or predicted column names and dist column together
    table_sorted = table.merge(train).sort_values(by = 'dist')
    neighbors = table_sorted.head(k)
    return neighbors

# k-nearest neighbor classifier, prints summary of prediction accuracy
#
# arguments
#   - training: dataframe of training data (4 folds for our 5-fold case)
#   - test: holdout fold or tuning set we're going to predict
#   - label: class or predicted column name
#   - k: number of neighbors we will select
#
# returns
#   - none
def knn_classifier(train, test, label, k):
    incorrect = 0
    correct = 0
    tied = 0

    for _, row in test.iterrows():
        neighbors = _find_knn(train, row, label, k)
        prediction = majority_vote(neighbors, label)

        if prediction == row[label]:
            correct += 1
        else:
            incorrect += 1
        
    total = incorrect + correct + tied

    print('-------- KNN SUMMARY --------')
    print('k =', k)
    print('# total\t\t', total)
    print('# correct:\t', correct)
    print('# incorrect:\t', incorrect)
    print('# tied:\t\t', tied)

# condensed k-nearest neighbor
# example set begins as empty and is added to iteratively
#       initialize set Z = { empty }, is subset of X
#           for every point random px in X:
#               find point in Z that is minimal distance to px
#               if classes do not agree, add px to Z
#
#   when Z is empty, add first point to Z and let it be misclassified
#   repeat over X several times until Z does not change
#
# arguments
#   - training: dataframe of training data (4 folds for our 5-fold case)
#   - test: whichever fold or tuning set we're going to predict
#   - label: class or predicted column name
#   - threshold: optional threshold for regression data
# returns
#   - condensed training set
def condensed_knn(train, test, label, threshold = None):
    z = pd.DataFrame()
    rounds = 2

    for i in range(rounds):
        l = len(z)
        z = _condense_helper(train, z, label)
        
        # z is still changing, continue
        if(len(z) > l):
            continue
    
    return z

# wrapper for inner loop to create Z in cnn
# arguments
#   - training
#   - z
#   - label: class or predicted column name
#   - threshold: optional threshold for regression data
#
# returns
#   - z: modified (growing) training set
def _condense_helper(train, z, label, threshold = None):
    for _, row in train.sample(frac = 1).iterrows():
        if (len(z) == 0):
            z = z.append(row)
        else:
            table = build_distance_table(z.drop(columns = label), row.drop(labels = label))
            train_subset = train.loc[table.index]
            table_sorted = table.merge(train_subset).sort_values(by = 'dist')
            prediction = table_sorted.head(1)[label].values

            # classifier: add points that do not match target
            if (prediction != row[label]):
                z = z.append(row)
            else:
                pass
            
            # regressor: add points outside threshold for target
            if (abs(prediction - row[label]) >= threshold):
                z = z.append(row)
            else:
                pass
    return z

# wrapper for inner loop to create Z in edited nearest neighbor
# arguments
#   - training
#   - z
#   - label: class or predicted column name
#   - threshold: optional threshold for regression data
#
# returns
#   - z: modified (reduced) training set
def _edit_helper(train, z, label, threshold = None):
    for _, row in train.sample(frac = 1).iterrows():
        z_x = z.drop(_)
        table = build_distance_table(z_x.drop(columns = label), row.drop(labels = label))
        train_subset = train.loc[table.index]
        table_sorted = table.merge(train_subset).sort_values(by = 'dist')
        prediction = table_sorted.head(1)[label].values

        # classifier: drop points that match our target
        if threshold == None:
            if (prediction == row[label]):
                z = z.drop(_)
            else:
                pass # keep
        
        # regressor: drop points outside threshold of target
        else:
            if (abs(prediction - row[label] >= threshold)):
                z = z.drop(_)
            else:
                pass # keep     

    return z

# edited k-nearest neighbor, begins with complete set of X examples
#
# start with full training set X
# randomly pick 1 point from X
# classify it against the remainder of X
#   - incorrect: remove from training set
#   - correct: keep in set
# at the end, we'll have a modified X' that is a subset of X, which we'll use for training
#
# arguments
#   - training: dataframe of training data (4 folds for our 5-fold case)
#   - test: whichever fold or tuning set we're going to predict
#   - label: class or predicted column name
#   - threshold: threshold, only used for regression data
#
# returns
#   - modified dataframe
def edited_knn(train, test, label, threshold = None):
    z = train.copy()
    z = _edit_helper(train, z, label, threshold)
    return z

# k-nearest neighbor regressor, prints parameters and mean-squared error (MSE)
#
# arguments
#   - training: dataframe of training data (4 folds for our 5-fold case)
#   - test: holdout fold or tuning set we're going to predict
#   - label: class or predicted column name
#   - k: number of neighbors we will select
#   - sigma: binning size
#
# returns
#   - None
def knn_regressor(train, test, label, k, sigma):
    train_dist = train.copy()

    acc = 0
    actual = []
    predicted = []

    for _, query in test.iterrows():
        acc = 0
        neighbors = _find_knn(train, query, label, k)
        for _, n in neighbors.iterrows():
            ki = rbf(n.drop(labels = ['dist']), query, sigma)
            acc += ki * n[label]

        actual.append(query[label])
        predicted.append(acc)

    mse = np.mean((np.subtract(actual, predicted)) ** 2)
    print('k\t', k)
    print('sigma\t', sigma)
    print('MSE\t', mse)