#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

m = 1
p = 0.0001

def build_probability_table(df, label):
    n = df.shape[0]

    ## create data frame to store probabilities
    ## dimensions = # class_opts x (# features + 2)
    ## each column = P(f = 0 | c), since the symmetrical f-value (f = 1) will just be [1 - P(f = 0 | c)]
    probdf = pd.DataFrame()

    cols = df.copy().drop(columns = label).columns.tolist()
    class_opts = df[label].value_counts()
    probdf = pd.DataFrame(0, index = class_opts.index, columns = cols)
    
    for c in class_opts.index:
        df2 = df.copy()
        df2 = df2[df2[label] == c]

        for f in cols:
            # probability that feature = 0
            # proability that feature = 1 will just be 1 - p0
            p0 = (df2[f].values == 0).sum()
            probdf.loc[c,f] = p0 / n

    return probdf