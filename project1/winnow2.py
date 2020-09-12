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

theta = 0.5
alpha = 2

def weighted_sum(row, weights):
    accum_sum = 0
    for idx, val in enumerate(row.to_numpy()):
        accum_sum += val * (weights[idx])
    return accum_sum

def promote(row, weights, alpha):
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
            new_weights[idx] = weights[idx] * alpha

    return new_weights

# divide weights of attr = 1
def demote(row, weights, alpha):
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
           new_weights[idx] = weights[idx] / alpha

    return new_weights

def build_classifier(df, label):
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

def build_classifier_multinomial(df, label):
    class_cols = [col for col in df if col.startswith(label)]
    class_cols = np.array(class_cols)
    classifiers = []
    for c in class_cols:
        drop_cols = np.setdiff1d(class_cols, [c])
        df2 = df.copy().drop(columns = drop_cols)
        out = build_classifier(df2, c)
        classifiers.append({'label': c, 'weights': out})
    
    return classifiers
    
def test_model(df, weights, label):
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0

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
    print('-- prediction for class: ', label)
    print('------------------')
    print('True +\t', true_pos)
    print('False +\t', false_pos)
    print('True -\t', true_neg)
    print('False -\t', false_neg)

def test_model_multinomial(df, label, classifiers):
    class_cols = [col for col in df if col.startswith(label)]
    class_cols = np.array(class_cols)
    for c in classifiers:
        drop_cols = np.setdiff1d(class_cols, [c['label']])
        df2 = df.copy().drop(columns = drop_cols)
        test_model(df2, c['weights'], c['label'])