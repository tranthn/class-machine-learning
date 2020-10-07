#!/usr/bin/env python3
import sys
class Node():
    def __init__(self, feature = None, transition_value = None, split_fn = None, items = None, decision = None):
        self.feature = feature
        self.split_fn = split_fn
        self.children = list()
        self.decision = decision
        self.transition_value = transition_value
        self.items = items

    def print(self, levels = 0):
        pre = '\t' * levels
        print()
        print('{0} feat: {1}'.format(pre, self.feature))
        print('{0} transition value: {1}'.format(pre, self.transition_value))
        if not (self.items is None):
            print('{0} items #: {1}'.format(pre, len(self.items)))

        if (len(self.children) > 0):
            print('{0} children #: {1}'.format(pre, len(self.children)))
            for c in self.children:
                c.print(levels = levels + 1)
        else:
            print(pre, 'leaf node')
            print(pre, 'decision = {0}'.format(self.decision))

    def append_child(self, node):
        self.children.append(node)