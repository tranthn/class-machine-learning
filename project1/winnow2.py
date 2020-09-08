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
    row = row.drop('class') # class does not factor into weights
    for idx, val in enumerate(row.to_numpy()):
        acc += val * (weights[idx])
    return acc

# def promote(row, weights, alpha):

# divide weights of attr = 1
def demote(row, weights, alpha):
    row = row.drop('class')
    new_weights = weights.copy()
    for idx, val in enumerate(row.to_numpy()):
        if (val == 1):
           new_weights[idx] = weights[idx] / alpha
    
    return new_weights

def build_table(df):
    label = 'class'
    n = len(df.columns.tolist()) - 1
    weights = [1.0] * n

    for _, row in df.iterrows():
        ws = weighted_sum(row, weights)

        # h(x) = 0
        if (ws <= theta):
            if row[label] == 1:
                weights = demote(row, weights, alpha)
        
        # h(x) = 1
        else:
            if row[label] == 0:
                print('false negative, demote')
                weights = demote(row, weights, alpha)
    
    print(weights)
