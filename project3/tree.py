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

# binary split for numeric columns
#
# returns
#   - median point
def get_numeric_split(df, column):
    print('\n- numeric split -')
    print('col\t', column)
    sorted = df.sort_values(by = column)
    md1 = int(df.shape[0] / 2)
    md2 = int(md1 + 1)

    values = sorted[column].to_numpy()
    split = (values[md1] + values[md2]) / 2
    print('split\t', split)

    med = df[column].median()
    print('median\t', med)

    return med

# calculate individual information gain value for given position / total combo
#
# arguments
#   - total: total # items in a given feature-attr combo
#   - pos: # of class positive examples with given feature-attr combo
#
# returns
#   - calculated log2 information gain
def info_gain(total, pos):
    # negative count is just remainder
    neg = total - pos
    p_pn = pos / total
    n_pn = neg / total

    if (p_pn == 0 or n_pn == 0):
        return 0
    else:
        i = (-p_pn * math.log(p_pn, 2)) - (n_pn * math.log(n_pn, 2))
        return i

# helper that calculates information gain for numeric columns
def entropy_helper_numeric(df, feature, class_label):
    print('\n---- entropy helper ----')
    tot = df.shape[0]
    split_point = get_numeric_split(df, feature)
    left = df[df[feature] <= split_point].count()[feature]
    left_pos = df[(df[feature] <= split_point) & (df[class_label] == 1)].count()[feature]

    right = df[df[feature] > split_point].count()[feature]
    right_pos = df[(df[feature] > split_point) & (df[class_label] == 1)].count()[feature]

    print('# <= {0}:\t {1}'.format(split_point, left))
    print('# <= {0} (p):\t{1} '.format(split_point, left_pos))
    print('# >> {0}:\t {1}'.format(split_point, right))
    print('# >> {0} (p):\t{1}'.format(split_point, right_pos))

    left_gain = (left / tot) * info_gain(left, left_pos)
    right_gain = (right / tot) * info_gain(right, right_pos)

    ent = right_gain + left_gain
    return ent

# wrapper that calculates entropy for a given feature
def entropy(df, feature, class_label):
    tot = df.shape[0]
    ent = 0
    ftype = df[feature].dtype

    if (ftype != 'float64' and ftype != 'int64'):
        # group just by feature-attr summary
        grouping = df.groupby(by = [feature], dropna = False).size()
        
        # group by feature-attr x class combos
        nested_grouping = df.groupby(by = [feature, class_label], dropna = False).size()
        feature_opts = grouping.index
        
        for f in feature_opts:

            # check if the feature has a split for predicting class or not
            # if the length = 0, that means this feature-opt combo all fell
            # within the same outcome class, take else branch
            if (len(nested_grouping[f]) > 1):
                p = nested_grouping[f, 1]
                pn = grouping[f]
                if_gain = info_gain(pn, p)
                ent += (pn / tot) * if_gain
            else:
                # gain should be 0
                if_gain = 0
    
    # handle numeric columns with separate split logic
    else:
        ent = entropy_helper_numeric(df, feature, class_label)

    print('ent\t', ent)
    return ent

# find best feature to be root node
def pick_feature(df, class_label):
    tot = df.shape[0]
    cols = df.columns.drop(class_label)

    # calculate set entropy
    grouping = df.groupby(by = [class_label]).size()
    set_entropy = info_gain(tot, grouping[1])
    print('\nset entropy', set_entropy)
    best_feat = ""
    best_gain = -1

    for f in cols:
        print('----------')
        print('f', f)
        e = entropy(df, f, class_label)
        total_gain = set_entropy - e
        if (total_gain > best_gain):
            best_gain = total_gain
            best_feat = f

    print('-- BEST FEATURE --')
    print(best_feat, best_gain)
    return best_feat

def id3_tree(df, class_label, tree = None):
    if (tree == None):
        tree = ID3Tree()

    root_candidate = pick_feature(df, class_label)

######################################################
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