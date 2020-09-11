#!/usr/bin/env python3
import sys
import csv
import numpy as np
import pandas as pd

m = 1
p = 0.0001

def build_probability_table(df, label, is_multinomial = False):
    n = df.shape[0]

    ## create data frame to store probabilities
    ## dimensions = # classes x (# features + 2)
    ## each column = P(f = 0 | c), since the symmetrical f-value (f = 1) will just be [1 - P(f = 0 | c)]
    probdf = pd.DataFrame()

    if (is_multinomial == False):
        
        cols = df.copy().drop(columns = label).columns.tolist()
        classes = df[label].value_counts()
        probdf = pd.DataFrame(0, index = classes.index, columns = cols)
        
        for c in classes.index:
            for f in cols:
                df2 = df.copy()
                # probability that feature = 0
                # proability that feature = 1 will just be 1 - p0
                p0 = (df2[f].values == 0).sum()
                probdf.loc[c,f] = p0 / n
    
    else:
        classes = [col for col in df if col.startswith(label)]
        classes = np.array(classes)
        cols = df.copy().drop(columns = classes).columns.tolist()

        print(classes)

        probdf = pd.DataFrame(0, index = classes, columns = cols)
        
        for c in classes:
            for f in cols:
                df2 = df.copy()
                # probability that feature = 0
                # proability that feature = 1 will just be 1 - p0
                p0 = (df2[f].values == 0).sum()
                probdf.loc[c,f] = p0 / n

    # print(probdf)