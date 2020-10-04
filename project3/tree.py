#!/usr/bin/env python3
import sys
import math
from anytree import NodeMixin, RenderTree
import numpy as np
import pandas as pd

## id3
## 1: figure out how to split numeric fields
##   - might be easiest to start with binary splits only
## 2: once splits can be determined, then we can determine information gain
## 3: main split knowledge as we build tree - maybe implement or use simple tree lib?
def get_numeric_split(df, column):
    print('\n-- get numeric split --')
    sorted = df.sort_values(by = column)
    md1 = int(df.shape[0] / 2)
    md2 = int(md1 + 1)

    values = sorted[column].to_numpy()
    split = (values[md1] + values[md2]) / 2
    print('split\t', split)

    med = df[column].median()
    print('median\t', med)

    return med

def pick_feature(df, class_label):
    tot = df.shape[0]
    cols = df.columns.drop(class_label)

    # calculate set entropy
    grouping = df.groupby(by = [class_label]).size()
    set_entropy = gain(tot, grouping[1])
    print('\nset entropy', set_entropy)

    for f in cols:
        print()
        print(f)
        e = entropy(df, f, class_label)
        total_gain = set_entropy - e
        print('total gain', total_gain)

def entropy(df, feature, class_label):
    tot = df.shape[0]
    ent = 0

    grouping = df.groupby(by = [feature], dropna = False).size()
    nested_grouping = df.groupby(by = [feature, class_label], dropna = False).size()
    feature_opts = grouping.index

    for f in feature_opts:
        # check if the feature has a split for predicting class or not
        # if the length = 0, that means this feature-opt combo all fell
        # within the same outcome class, take else branch
        if (len(nested_grouping[f]) > 1):
            p = nested_grouping[f, 1]
            pn = grouping[f]
            if_gain = gain(pn, p)
            ent += (pn / tot) * if_gain
        else:
            # gain should be 0
            if_gain = 0
    
    print('ent', ent)
    return ent

def gain(total, pos):
    # negative count is just remainder
    neg = total - pos
    p_pn = pos / total
    n_pn = neg / total
    i = (-p_pn * math.log(p_pn, 2)) - (n_pn * math.log(n_pn, 2))
    return i

class ID3Tree(object):
    foo = 4

class ID3TreeNode(ID3Tree, NodeMixin):
    def __init__(self, name, items, split, parent=None, children=None):
        super(ID3Tree, self).__init__()
        self.name = name
        self.item = items
        self.split = split
        self.parent = parent
        if children:  # set children only if given
            self.children = children