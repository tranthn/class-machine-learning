#!/usr/bin/env python3
import sys
import math
import anytree
import numpy as np
import pandas as pd

## id3
## 1: figure out how to split numeric fields
##   - might be easiest to start with binary splits only
## 2: once splits can be determined, then we can determine information gain
## 3: main split knowledge as we build tree - maybe implement or use simple tree lib?
def get_numeric_split(data, column):
    print('\n-- get numeric split --')
    sorted = data.sort_values(by = column)
    md1 = int(data.shape[0] / 2)
    md2 = int(md1 + 1)

    values = sorted[column].to_numpy()
    split = (values[md1] + values[md2]) / 2
    print('split\t', split)

    med = data[column].median()
    print('median\t', med)

    return med

# id3 implementation 
#
# classification: gain ratio for splitting criteria
# regression: split sorted data and do binary splits at midpoints between adajacent data nodes
# 
# arguments
#   - data
#   - pruning: whether or not to use reduced error pruning [classification]
# def id3_tree(data, pruning = False):
#     pass