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

    # print('ent\t', ent)
    return ent

# find best feature to be root node
def pick_feature(df, class_label):
    tot = df.shape[0]
    cols = df.columns.drop(class_label)

    # calculate set entropy
    grouping = df.groupby(by = [class_label]).size()
    set_entropy = info_gain(tot, grouping[1])
    # print('\nset entropy', set_entropy)
    best_feat = ""
    best_gain = -1

    for f in cols:
        e = entropy(df, f, class_label)
        total_gain = set_entropy - e
        if (total_gain > best_gain):
            best_gain = total_gain
            best_feat = f

    return best_feat

def id3_tree(df, label, tree = None, features = None):
    print('---------------------\n')
    print('remaining features', features)
    print('df\n', df)

    root_node = ID3Node(items = df)

    # if tree hasn't started yet, set to current root
    if (tree == None):
        tree = root_node

    # label decision on node
    if (len(features) <= 1):
        print('remaining features <=1')    
        return tree

    # create root
    root = pick_feature(df, label)
    root_node.feature = root

    # recalculate groupings to create children nodes
    grouping = df.groupby(by = [root], dropna = False).size()
    feature_opts = grouping.index

    # make leaf node with decision
    if (len(feature_opts) == 1):
        print('only 1 feature option for ', feature_opts)
        return tree

    # there are more than 1 features left
    print('\nroot', root)
    print(feature_opts.values)
    next_features = np.delete(features, np.where(features == [root]))

    for f in feature_opts.values:
        # only 1 class represented, so it becomes leaf
        nested_grouping = df.groupby(by = [root, label], dropna = False).size()

        if (len(nested_grouping[f]) == 1):
            print('feature-attr combo is all 1 class')
            leaf = ID3Node(feature = root, decision = nested_grouping[f])
            tree.append_child(leaf)
        else:
            # get df split for given feature-attr
            subset = df[df[root] == f]
            subset = subset.drop(columns = root)

            # recursive call with given feature-attr split
            print('\nrecursive call')
            print('next features', next_features)
            subtree = id3_tree(subset, label, tree, next_features)
            tree.append_child(subtree)

    return tree

######################################################
class ID3Node():
    def __init__(self, feature = None, split_fn = None, items = None, decision = None):
        self.feature = feature
        self.split_fn = split_fn
        self.children = []
        self.items = items

    def __str__(self):
        pretty = "feature\n{0}\n, items\n{1}".format(self.feature, self.items)
        return pretty

    def append_child(self, node):
        self.children.append(node)