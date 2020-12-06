#!/usr/bin/env python3
import numpy as np
import random
from termcolor import colored, cprint

# velocity options range from -5 to 5
# uses negative indexing for easier coding, but it can be a little confusing, see explanation below
"""
real index:      0   1   2   3   4   5   6   7   8   9   10
index used:      0   1   2   3   4   5   -5  -4  -3  -2  -1

so if I want value for a cell with velocity [-1, -5], it will work but its true indices in printout would be [10, 6]
"""

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
    def __init__(self, env = None, vl_opts = [-1, 0, 1], actions = [(0,0), (0,1), (1,0)],
                        gamma = 1.0, epsilon = 0.1):
        # TrackSimulator object
        self.env = env
        self.vl_opts = vl_opts
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon

    def _print_vtable(self, row):
        print('vr, vc')
        for vr in self.vl_opts:
            for vc in self.vl_opts:
                print('({0},{1})\t\t'.format(vr, vc), end = '')
                for col in range(self.env.ncols()):
                    print(row[col, vr, vc], '\t', end = '')
                print()

    def _print_qtable(self, row):
        print('vr, vc')
        for vr in self.vl_opts:
            for vc in self.vl_opts:
                print('({0},{1})\t\t'.format(vr, vc))
                for a in range(len(self.actions)):
                    for col in range(self.env.ncols()):
                        print('\t', self.actions[a], end = '')
                        print('=', end = '')
                        print(row[col, vr, vc, a], ' ', end = '')
                    print()
                print()

    def pretty_print_table(self, table):
        if (len(table.shape) == 4):
            print('\t', end = '')
            for c in range(self.env.ncols()):
                print('\t  ', c, end = '')
            print()
            for r in range(self.env.nrows()):
                row = table[r]
                print(r)
                self._print_vtable(row)
                print('\n')
        elif(len(table.shape) == 5):
            # print('\t', end = '')
            for c in range(self.env.ncols()):
                print('\t  ', c, end = '')
            print()
            for r in range(self.env.nrows()):
                row = table[r]
                print(r)
                self._print_qtable(row)
                print('\n')

################################################################################
    def initialize(self):
        self.initialize_state_table()
        self.initialize_q_table()

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
        vr_opts = vc_opts = len(self.vl_opts) # values range from -5 to 5

        # initialilize reward to -1 for all entries
        table = np.full((r, c, vr_opts, vc_opts), -1)

        # set reward values for F / finish states to 0
        finish_pts = self.env.get_all_points_of(target = 'F')
        for pts in finish_pts:
            table[pts[0], pts[1]] = 0
        
        self.vtable = table

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
        vr_opts = vc_opts = len(self.vl_opts) # values range from -5 to 5
        act_opts = len(self.actions)
        table = np.full((r, c, vr_opts, vc_opts, act_opts), -1)

        # set reward values for F / finish states to 0
        finish_pts = self.env.get_all_points_of(target = 'F')
        for pts in finish_pts:
            table[pts[0], pts[1]] = 0
        
        self.qtable = table

    def value_iteration(self):
        self.initialize()
        rows = self.env.nrows()
        cols = self.env.ncols()
        vopts = self.vl_opts
        track = self.env.track
        threshold = 0.8
        epsilon = self.epsilon
        gamma = self.gamma

        # 4-nested for loop to iterate through all state combinations
        # S = (r, c, vr, vc, reward)
        for r in range(rows):
            for c in range(cols):
                for vr in vopts:
                    for vc in vopts:
                        # penalize wall states and move on to next state
                        if track[r, c] == '#':
                            self.vtable[r, c, vr, vc] = -10
                            continue
                    
                        # on a given state, we'll look at available actions
                        for aidx, action in enumerate(self.actions):
                            # set current running reward, 0 if on finish cell
                            reward = 0 if (track[r, c] == 'F') else -1

                            # get current position/coordinates
                            vel1 = self.env.velocity
                            pos1 = self.env.position

                            # get value if no acceleration change
                            pos_no_acc = self.env.move()
                            vel_no_acc = self.env.velocity

                            # take current action on track simulator
                            # accelerate(): internally handles chance of acceleration failure and adjusts velocity accordingly
                            # move(): handles updating position and restarting if crashed
                            vel2 = self.env.accelerate(action[0], action[1])
                            pos2 = self.env.move()
                            self.env.finalize_move()

                            # compare values
                            val1 = self.vtable[pos1[0], pos1[1], vel1[0], vel1[1]]
                            val2 = self.vtable[pos2[0], pos2[1], vel2[0], vel2[1]]
                            val_no_acc = self.vtable[pos_no_acc[0], pos_no_acc[1], vel_no_acc[0], vel_no_acc[1]]

                            # take transition probabilities into account
                            new_value = (1 - threshold) * val_no_acc + threshold * val2
                            self.qtable[r, c, vr, vc, aidx] = reward + gamma * new_value 

                            # determine which action had highest q-value, to use to set value
                            act_maxq = np.argmax(self.qtable[r, c, vr, vc])
                            maxq = self.qtable[r, c, vr, vc, act_maxq]
                            self.vtable[r, c, vr, vc] = maxq
        
        self.pretty_print_table(self.vtable)