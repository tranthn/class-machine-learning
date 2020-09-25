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

# calculate gaussian radial basis function
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
    k = dist * gamma

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
#  returns None is there is a tie between top 2 class predictions
#
# arguments
#   - neighbors: k-neighbors, # passed in determined by caller
#   - label: class label
#
# returns
#   - predicted class
def majority_vote_with_ties(neighbors, label):
    # value_counts() will return in descending count by default
    nn = neighbors[label].value_counts()
    best_neighbor = nn.index[0]
    second_neighbor = None

    # assign second neighbor's class, if it exists
    if (len(nn.index) > 1):
        second_neighbor = nn.index[1]

    if second_neighbor != None:
        # second neighbor exists, but they don't have same count
        if nn.values[0] != nn.values[1]:
            return best_neighbor
        # second neighbor exists, tie condition
        else:
            return None
    else:
        return best_neighbor

#  determines which class is the majority among the picked neighbors
#  allows ties and just picks first option
#
# arguments
#   - neighbors: k-neighbors, # passed in determined by caller
#   - label: class label
#
# returns
#   - predicted class
def majority_vote(neighbors, label):
    # value_counts() will return in descending count by default
    nn = neighbors[label].value_counts()
    best_neighbor = nn.index[0]

    return best_neighbor

# main entry point for calculating k-nearest neighbor
#
# arguments
#   - training: array of dataframes representing our current 4-set of folds
#   - test: whichever fold or tuning set we're going to predict
#   - label: class label
#   - k: number of neighbors we will select
#
# returns
#   - none
def knn_classifier(train, test, label, k):
    incorrect = 0
    correct = 0
    tied = 0

    for _, row in test.iterrows():
        table = build_distance_table(train.drop(columns = label), row.drop(labels = label))

        # merge back to ensure the table has both the class labels and dist column together
        table_sorted = table.merge(train).sort_values(by = 'dist') 
        prediction = majority_vote(table_sorted.head(k), label)

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
#   - training: array of dataframes representing our current 4-set of folds
#   - test: whichever fold or tuning set we're going to predict
#   - label: class label
#   - k: number of neighbors we will select
#
# returns
#   - condensed training set
def condensed_knn(train, test, label):
    z = pd.DataFrame()
    rounds = 5

    for i in range(rounds):
        l = len(z)
        z = condense_helper(train, z, label)
        
        # z is still changing, continue
        if(len(z) > l):
            continue
    
    return z

# wrapper for inner loop to create Z in cnn
# arguments
#   - training
#   - z
def condense_helper(train, z, label):
    for _, row in train.sample(frac = 1).iterrows():
        if (len(z) == 0):
            z = z.append(row)
        else:
            table = build_distance_table(z.drop(columns = label), row.drop(labels = label))
            train_subset = train.loc[table.index]
            table_sorted = table.merge(train_subset).sort_values(by = 'dist')
            prediction = majority_vote(table_sorted.head(1), label)

            if (prediction == row[label]):
                pass
            else:
                z = z.append(row)
    return z

# wrapper for inner loop to create Z in edited nearest neighbor
# arguments
#   - training
#   - z
def edit_helper(train, z, label):
    for _, row in train.sample(frac = 1).iterrows():
        z_x = z.drop(_)
        table = build_distance_table(z_x.drop(columns = label), row.drop(labels = label))
        train_subset = train.loc[table.index]
        table_sorted = table.merge(train_subset).sort_values(by = 'dist')
        prediction = majority_vote(table_sorted.head(1), label)

        if (prediction != row[label]):
            pass
        else:
            z = z.drop(_)

    return z

# edited k-nearest neighbor, begins with complete set of X examples
# removing examples can improve accuracy

# start with full training set X
# randomly pick 1 point from X
# classify it against the remainder of X
#   - incorrect: remove from training set
#   - correct: keep in set
# at the end, we'll have a modified X' that is a subset of X, which we'll use for training

# arguments
#   - training: array of dataframes representing our current 4-set of folds
#   - test: whichever fold or tuning set we're going to predict
#   - label: class label
#   - k: number of neighbors we will select

# returns
#   - modified dataframe
def edited_knn(train, test, label):
    z = train.copy()
    z = edit_helper(train, z, label)
    return z

def knn_regressor(train, test, label, sigma):
    train_dist = train.copy()
    point = test.head(1).drop(columns = label)
    x = train_dist.head(1).drop(columns = label)

    print(point)
    print(x)
    rbf(point, x, sigma)