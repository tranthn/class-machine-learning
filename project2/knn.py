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

# calculates euclidean distance given 2 pandas dataframe rows
#
# arguments
#   - v1: row vector representing a point
#   - v2: row vector representing a point
#
# returns
#   - unrounded euclidean distance
def euclidean_dist(v1, v2):
    square = lambda x : x ** 2

    # get array representing pairwise subtraction of vector elements
    v = np.subtract(v1, v2)

    # apply our lambda fn to square all values
    arr = np.array([square(vi) for vi in v])
    dist = math.sqrt(np.sum(arr))

    return dist

#  modifies dataframe to add calculated distance column w.r.t target point
#
# arguments
#   - training: data frame representing our training rows
#   - point: target instance we're going to predict class for, via euclidean distance
#
# returns
#   - modified dataframe
def build_distance_table(training, point):
    trains_dist = training.copy()
    trains_dist['dist'] = training.apply(lambda x : euclidean_dist(point, x), axis = 1)
    return trains_dist

#  determines which class is the majority among the picked neighbors
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
    second_neighbor = None
    k = len(neighbors)

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

# main entry point for calculating k-nearest neighbor
#
# arguments
#   - training: array of dataframes representing our current 4-set of folds
#   - test: whichever fold or tuning set we're going to predict
#   - label: class label
#   - k: number of neighbors we will select
#
# returns
#   - modified dataframe
def find_knn(train, test, label, k):
    trains = pd.concat(train)
    incorrect = 0
    correct = 0
    tied = 0

    for _, row in test.iterrows():
        table = build_distance_table(trains.drop(columns = label), row.drop(labels = label))
        table_sorted = table.merge(trains).sort_values(by = 'dist') 
        prediction = majority_vote(table_sorted[0:k], label)

        if prediction == None:
            tied += 1
        elif prediction == row[label]:
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