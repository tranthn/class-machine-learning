#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

m = 1
p = 0.0001

def build_probability_table(df):
    label = 'class'
    n = df.size

    ## create data frame to store probabilities
    ## dimensions = # classes x (# features + 2)
    ## each column = P(f = 0 | c), since the symmetrical f-value (f = 1) will just be [1 - P(f = 0 | c)]
    
    df2 = df.copy()
    df2 = df2.groupby([label]).size()

    cols = df.copy().drop(columns = label).columns.tolist()
    cols.insert(0, label)
    indices = np.arange(len(cols) + 1)

    pdf = pd.DataFrame(0, index = indices, columns = cols)
    print(df2)
    # pdf[:, 0] = df.apply(lambda x: x / n)
    df2 = df2.apply(lambda x: x / n)
    print(df2)

    for f in cols:
        df2 = df.copy()
        df2 = df2.groupby([label, f]).size()
        print(df2)

        """
        df2[0,0] # c = 0, feat = 0
        df2[0,1] # c = 0, feat = 1
        df2[1,0] # c = 1, feat = 0
        df2[1,1] # c = 1, feat = 1
        """
        