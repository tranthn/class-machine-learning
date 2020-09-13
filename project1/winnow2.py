#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

## winnow steps
## with training set, experiment with
##  - weight vectors: init to 1, for each feature
##  - theta: threshold, usually = 0.5
##      -- but also n / 2 is good
##  - alpha: modifier (> 1)
theta_default = 0.5
alpha_default = 2

# calculates the weighted sum for a given data row
# for every feature column = 1, we will add its
# corresponding weight to the accumulated sum
#
# arguments
#   - row = a given instance of data, whose values we will use to modify the weights
#   - alpha = the modifer we will use to demote the weights
#
# returns
#   - new_weights = weight array after demotion process
def weighted_sum(row, weights):
    accum_sum = 0
    for idx, val in enumerate(row.to_numpy()):
        accum_sum += val * (weights[idx])
    return accum_sum

# handles the promotion process for our weights
# occurs when a given model predicts class exclusion incorrectly
#
# arguments
#   - row = a given instance of data, whose values we will use to modify the weights
#   - weights = original weights that we will modify
#   - alpha = the modifer we will use to promote the weights
#
# returns
#   - new_weights = weight array after promotion process
def promote(row, weights, alpha):
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
            new_weights[idx] = weights[idx] * alpha

    return new_weights

# handles the demotion process for our weights
# occurs when a given model predicts class belonging incorrectly
#
# arguments
#   - row = a given instance of data, whose values we will use to modify the weights
#   - weights = original weights that we will modify
#   - alpha = the modifer we will use to demote the weights
#
# returns
#   - new_weights = weight array after demotion process
def demote(row, weights, alpha):
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
           new_weights[idx] = weights[idx] / alpha

    return new_weights

# creates weight array representing our winnow2 model
#
# arguments
#   - df = dataframe we will be using as our training set
#   - label = name of the column of dataframe that maps to the label/class
#
# returns
#   - classifers: array of dict-objects, mapping each weight array to a classifer choice
def build_classifier(df, label, theta = theta_default, alpha = alpha_default):
    num_cols = len(df.columns.tolist()) - 1
    weights = [1.0] * num_cols

    for _, row in df.iterrows():
        row_nc = row.drop(labels = [label])
        ws = weighted_sum(row_nc, weights)

        # predicted h(x) = 0
        if (ws <= theta):
            if row[label] == 1:
                # print('false negative, demote')
                weights = promote(row_nc, weights, alpha)
        
        # predicted h(x) = 1
        else:
            if row[label] == 0:
                # print('false positive, demote')
                weights = demote(row_nc, weights, alpha)
    
    return weights

# handles multinomial data, i.e. data where an instance has 3+ options for its label
# this wraps around the build_classifer() method above, creating a classifier
# for each label option, treating that class as its own 2-class problem
#
# arguments
#   - df = dataframe we will be using as our training set
#   - label = name of the column of dataframe that maps to the label/class
#
# returns
#   - classifers: array of dict-objects, mapping each weight array to a classifer choice
def build_classifier_multinomial(df, label, theta = theta_default, alpha = alpha_default):
    class_cols = [col for col in df if col.startswith(label)]
    class_cols = np.array(class_cols)
    classifiers = []
    for c in class_cols:
        drop_cols = np.setdiff1d(class_cols, [c])
        df2 = df.copy().drop(columns = drop_cols)
        out = build_classifier(df2, c, theta, alpha)
        classifiers.append({'label': c, 'weights': out})
    
    return classifiers

# tests the classifer that has been built 
#
# arguments
#   - df = dataframe we will be using as our training set
#   - weights = the weights representing our model
#   - label = name of the column of dataframe that maps to the label/class
def test_model(df, weights, label, theta):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    total = df.shape[0]

    for _, row in df.iterrows():
        row_nc = row.copy().drop(label) # class does not factor into weights
        ws = weighted_sum(row_nc, weights)

        # predicted h(x) = 1
        if (ws <= theta):
            if row[label] == 1:
                false_neg += 1
            else:
                true_neg += 1
        
        # predicted h(x) = 1
        else:
            if row[label] == 0:
                false_pos += 1
            else:
                true_pos += 1

    print('\nWINNOW SUMMARY')
    print('prediction for class: ', label)
    print('------------------')
    print('Total #\t\t', total)
    print('--')
    print('True +\t\t', true_pos)
    print('False +\t\t', false_pos)
    print('True -\t\t', true_neg)
    print('False -\t\t', false_neg)
    print('--')
    print('Correct\t\t', (true_neg + true_pos))
    print('Wrong\t\t', (false_neg + false_pos))

# wrapper function, that tests classifers against a multinomial dataset 
#
# arguments
#   - df = dataframe we will be using as our training set
#   - label = name of the column of dataframe that maps to the label/class
#   - classifers: array of dict-objects, mapping each weight array to a classifer choice
def test_model_multinomial(df, classifiers, label, theta):
    class_cols = [col for col in df if col.startswith(label)]
    class_cols = np.array(class_cols)
    for c in classifiers:
        drop_cols = np.setdiff1d(class_cols, [c['label']])
        df2 = df.copy().drop(columns = drop_cols)
        test_model(df2, c['weights'], c['label'], theta)