#!/usr/bin/env python3
import numpy as np
from termcolor import colored, cprint

ACTIONS = [
    (-1,-1), (0,-1), (1,-1),
    (-1, 0), (0, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
]

# velocity options range from -5 to 5
# uses negative indexing for easier coding, but it can be a little confusing, see explanation below
"""
real index:      0   1   2   3   4   5   6   7   8   9   10
index used:      0   1   2   3   4   5   -5  -4  -3  -2  -1

so if I want value for a cell with velocity [-1, -5], it will work but its true indices in printout would be [10, 6]
"""

VL_OPTS = [0, 1,  2, 3, 4, 5, -5, -4, -3, -2, -1]
VL_OPTS = [0, 1, -1]

################################################################################

# sequence of V_t using Q_t(s, a)
#   - Q is auxillary function, represents estimate of value of action *a* taken on state *s*
#   - values of Q_t and V_t updated as we go along, until we drop below error threshold
#
# value iteration is essentially a form of dynamic programming

""" PSEUDOCODE
short hands
--------------------------------------------
    gm = discount factor
    epsilon = exploration factor, 0 = greedy, higher value = better results, slower performance
    R(s, a) = reward of state, action combo
    
    ss = s' in S, so all potential next states in S
    pi = policy
    aa = action option in A

actual pseudocode
--------------------------------------------
    for all states in S:
        V_0 = 0
    
    t = 0
    loop until max_(s in S) |V_t (s) - V_t-1(s) | < epsilon:
        t++
        for all states in S:
            for all actions in A:
                Q_t(s, a) = R(s, a) + gm * SUMMATION_ss [ T(s, a, s') * V_t-1 (s') ]
            
            pi_t(s) = argmax_aa Q_t(s, a)
            V_t(s) = Q_t (s, pi_t(s))
    return pi_t
"""

class ValueIteration():
    def __init__(self, env = None):
        # TrackSimulator object
        self.env = env

    def initialize(self):
        self.initialize_state_table()
        # self.initialize_q_table()

    def _print_helper(self, row):
        print('vr, vc')
        for vr in VL_OPTS:
            for vc in VL_OPTS:
                print('({0},{1})\t\t'.format(vr, vc), end = '')
                for col in range(self.env.ncols()):
                    print(row[col, vr, vc], '\t', end = '')
                print()

    def pretty_print_table(self, table):
        # Values table
        if (len(table.shape) == 4):
            print('\t', end = '')
            for c in range(self.env.ncols()):
                print('\t  ', c, end = '')
            print()
            for r in range(self.env.nrows()):
                row = table[r]
                print(r)
                self._print_helper(row)
                print('\n')

    def initialize_state_table(self):
        # this table needs to hold values for all our states
        # i.e. all positionals x all velocity combos
        # would have 4 dimensions to track:
        #   - r
        #   - c
        #   - vr - velocity along rows (y)
        #   - vc - velocity along columns (x)
        r = self.env.nrows()
        c = self.env.ncols()
        vr_opts = vc_opts = len(VL_OPTS) # values range from -5 to 5

        # initialilize reward to -1 for all entries
        table = np.full((r, c, vr_opts, vc_opts), -1)

        # set reward values for F / finish states to 0
        finish_pts = self.env.get_all_points_of(target = 'F')
        for pts in finish_pts:
            table[pts[0], pts[1]] = 0

        print('value table')
        self.pretty_print_table(table)

    def initialize_q_table(self):
        # this table needs to hold Q-values
        # i.e. all positionals x all velocity x action combos
        # would have 5 dimensions to track:
        #   - r
        #   - c
        #   - vr = velocity along rows (y)
        #   - vc = velocity along columns (x)
        #   - a = # possible actions, aka the acceleration options
        r = self.env.nrows()
        c = self.env.ncols()
        vr_opts = vc_opts = len(VL_OPTS) # values range from -5 to 5
        a = len(ACTIONS)
        table = np.full((r, c, vr_opts, vc_opts, a), -1)
        print('qtable')
        print(table.shape)
