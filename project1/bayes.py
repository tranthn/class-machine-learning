#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

m = 1
p = 0.001

def build_probability_table(df, label):
    tot = df.shape[0]

    ## create data frame to store probabilities
    ## dimensions = # class_opts x (# features + 2)
    probdf = pd.DataFrame()

    cols = df.copy().drop(columns = label).columns.tolist()
    class_opts = df[label].value_counts()
    probdf = pd.DataFrame(0, index = class_opts.index, columns = cols)
    probdf.insert(loc = 0, column = 'class%', value = 0)

    for c in class_opts.index:
        df2 = df.copy()

        # grab the rows with that match index == c (class option)
        df2 = df2[df2[label] == c]
        n = df2.shape[0]
        probdf.loc[c,'class%'] = n / tot

        for f in cols:
            # probability that feature = 1
            # proability that feature = 0 will just be 1 - pf
            pf = (df2[f].values == 1).sum()
            probdf.loc[c,f] = (pf + m * p) / (n + m)

    return probdf

## todo: factor in m-estimate to handle probability = 0
def compute_probability(instance, prob_arr):
    x = len(prob_arr)
    prob = 1.0
    if (len(instance) == x):
        for i in range(0, x - 1):
            # prob_arr[i] stores P(f = 1 | c)
            if (instance[i] == 1):
                prob = prob  * prob_arr[i]

            # 1 - prob_arr[i] gives us P(f = 0 | c)
            else:
                prob = prob * (1 - prob_arr[i])
    else:
        print("ARRAY LENGTH MISMATCH")

    return prob

def check_instance(row, probdf, class_opts):
    choice = None
    prob_max = 0.0
    for c in class_opts.index:
        class_per = probdf.loc[c, 'class%']
        probs = probdf.drop(columns = 'class%').to_numpy()[0]
        prob = class_per * compute_probability(row, probs)
        
        if (prob > prob_max):
            prob_max = prob
            choice = c

    return {'choice': c, 'probability': prob_max}

def test_model(df, probdf, label):
    class_opts = df[label].value_counts()
    for _, row in df.iterrows():
        row = row.drop(labels = label)
        outcome = check_instance(row, probdf, class_opts)
        print(outcome)