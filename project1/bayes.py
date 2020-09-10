#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

m = 1
p = 0.0001

def build_probability_table(df):
    label = 'class'
    n = df.shape[0]

    ## create data frame to store probabilities
    ## dimensions = # classes x (# features + 2)
    ## each column = P(f = 0 | c), since the symmetrical f-value (f = 1) will just be [1 - P(f = 0 | c)]
    df2 = df.copy()
    df2 = df2.groupby([label]).size()

    cols = df.copy().drop(columns = label).columns.tolist()
    classes = df['class'].value_counts()
    probdf = pd.DataFrame(0, index = classes.index, columns = cols)
    
    for c in classes.index:
        for f in cols:
            df2 = df.copy()
            c0 = (df2[f].values == 0).sum()
            c1 = (df2[f].values == 1).sum()
            probdf.loc[c,f] = c0 / n

    print(probdf)