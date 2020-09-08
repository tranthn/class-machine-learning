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
    acc = 0
    for idx, val in enumerate(row.to_numpy()):
        acc += val * (weights[idx])
    return acc

def promote(row, weights, alpha):
    row = row.copy().drop('class')
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
           new_weights[idx] = weights[idx] * alpha
    
    return new_weights

# divide weights of attr = 1
def demote(row, weights, alpha):
    row_nc = row.copy().drop('class')
    new_weights = weights.copy()
    for idx, val in enumerate(row_nc.to_numpy()):
        if (val == 1):
           new_weights[idx] = weights[idx] / alpha
    
    return new_weights

def build_table(df):
    label = 'class'
    n = len(df.columns.tolist()) - 1
    weights = [1.0] * n

    for _, row in df.iterrows():
        row_nc = row.copy().drop('class') # class does not factor into weights
        ws = weighted_sum(row_nc, weights)

        # predicted h(x) = 0
        if (ws <= theta):
            if row[label] == 1:
                print('false negative, demote')
                weights = promote(row, weights, alpha)
        
        # predicted h(x) = 1
        else:
            if row[label] == 0:
                print('false positive, demote')
                weights = demote(row, weights, alpha)
    
    return weights

def test_model(df, weights):
    label = 'class'
    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    print('---')
    print(df)

    for _, row in df.iterrows():
        row_nc = row.copy().drop('class') # class does not factor into weights
        ws = weighted_sum(row_nc, weights)

        # predicted h(x) = 0
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

    print('Summary statistics')
    print('------------------')
    print('True +\t', true_pos)
    print('False +\t', false_pos)
    print('True -\t', true_neg)
    print('False -\t', false_neg)