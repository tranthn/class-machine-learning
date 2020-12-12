#!/usr/bin/env python3
import numpy as np
import time
import random
import copy
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
short hand
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

global print_values
print_values = True

class ValueIteration():
    def __init__(self, env = None, vl_opts = [-1, 0, 1], 
                        actions = [(0,0), (0,1), (1,0)],
                        gamma = 1.0, epsilon = 0.1):
        # TrackSimulator object
        self.env = env
        self.vl_opts = vl_opts
        self.actions = actions
        self.gamma = gamma
        self.epsilon = epsilon

    # helper method to print v-table during debugging
    def _print_vtable(self, row):
        print('vr, vc')
        for vr in self.vl_opts:
            for vc in self.vl_opts:
                print('({0},{1})\t\t'.format(vr, vc), end = '')
                for col in range(self.env.ncols()):
                    print(row[col, vr, vc], '\t', end = '')
                print()

    # helper method to print q-table during debugging
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

    # helper to print values (4-d) or qtable (5-d)
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

    """
        initializes table to hold values

        i.e. all positionals x all velocity combos
        would have 4 dimensions to track:
            - r
            - c
            - vr - velocity along rows (y)
            - vc - velocity along columns (x)
    """
    def initialize_state_table(self):
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

    """
        initializes table to hold q-values

        i.e. all positionals x all velocity x action combos
        would have 5 dimensions to track:
            - r
            - c
            - vr = velocity along rows (y)
            - vc = velocity along columns (x)
            - a = # possible actions, aka the acceleration options
    """
    def initialize_q_table(self):
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

    """
        Main training method for ValueIteration
            - initialize value table
            - initializes Q-value table
            - runs training iterations and nested loop for finding optimal action
            - stops early if the max value-change is dropping below epsilon
            - returns finalized built policy at the end

        arguments:
            - iterations: default is 5

        returns:
            - policy: dictionary representing the optimal state-action combos
    """
    def train(self, iterations = 5):
        self.initialize()
        rows = self.env.nrows()
        cols = self.env.ncols()
        vopts = self.vl_opts
        track = self.env.track
        threshold = 0.8
        epsilon = self.epsilon
        gamma = self.gamma
        max_vchange = 100
        global print_values

        for i in range(iterations):
            vtable_prior = copy.deepcopy(self.vtable)
            if (i % 5 == 0):
                print('\nvalue iteration learning, #', i)

            # 4-nested for loop to iterate through all state combinations
            # S = (r, c, vr, vc)
            start = time.time()
            for r in range(rows):
                for c in range(cols):
                    for vr in vopts:
                        for vc in vopts:
                            # penalize wall states and move on to next state
                            if track[r, c] == '#':
                                self.vtable[r, c, vr, vc] = -9.9
                                continue
                        
                            # on a given state, we'll look at available actions
                            for aidx, action in enumerate(self.actions):
                                # set current running reward, 0 if on finish cell
                                reward = 0 if (track[r, c] == 'F') else -1

                                # set current position/coordinates based on loop values
                                self.env.position = pos1 = (r, c)
                                self.env.velocity = vel1 = (vr, vc)

                                # get value if no acceleration change
                                pos_no_acc = self.env.move()

                                # take current action on track simulator
                                # accelerate(): internally handles chance of acceleration failure and adjusts velocity accordingly
                                # move(): handles updating position and restarting if crashed
                                next_vel = self.env.accelerate(action[0], action[1])
                                next_pos = self.env.move()
                                self.env.finalize_move()

                                # compare values
                                # val_old = vtable_prior[pos1[0], pos1[1], vel1[0], vel1[1]]
                                val_new = vtable_prior[next_pos[0], next_pos[1], next_vel[0], next_vel[1]]
                                val_no_acc = vtable_prior[pos_no_acc[0], pos_no_acc[1], vel1[0], vel1[1]]

                                # take transition probabilities into account
                                future_reward = (1 - threshold) * val_no_acc + threshold * val_new
                                qvalue = reward + (gamma * future_reward)
                                self.qtable[r, c, vr, vc, aidx] = qvalue

                            # determine which action had highest q-value, to use to set value
                            act_maxq_idx = np.argmax(self.qtable[r, c, vr, vc])
                            maxq = self.qtable[r, c, vr, vc, act_maxq_idx]

                            if print_values:
                                print('V with no action\t', val_no_acc)
                                print('V with new action\t', val_new)
                                print_values = False

                            self.vtable[r, c, vr, vc] = maxq

            # set reward of finish states
            for r in range(rows):
                for c in range(cols):
                    if (track[r, c]) == 'F':
                        for vc in vopts:
                            for vr in vopts:
                                self.vtable[r, c, vr, vc] = 0

            # early break if the maximal state value change has dropped low enough
            max_vchange = self.find_max_vchange(abs(self.vtable - vtable_prior))
            if max_vchange < epsilon:
                print('max vchange has dropped below epsilon: ', max_vchange, epsilon)
                policy = self.build_policy(rows, cols, vopts)
                break

        # store final optimal action for a given state combo
        policy = self.build_policy(rows, cols, vopts)

        return policy

    """
    Helper that builds the policy based on the Q-value table
        - backtracks through all grid cells and velocity combos
        - finds the action with maximal q-value and picks that action
    """
    def build_policy(self, rows, cols, vopts):
        policy = {}
        start = time.time()
        for r in range(rows):
            for c in range(cols):
                for vr in vopts:
                    for vc in vopts:
                        act_maxq_idx = np.argmax(self.qtable[r, c, vr, vc])
                        best_action = self.actions[act_maxq_idx]
                        policy[(r,c,vr,vc)] = best_action

        print('policy backtrack took:\t{:.2f}s'.format(time.time() - start))
        return policy

    """
        calculates the highest value change in the value table before and after action

        arguments:
            - vtable_diff: absolute value of differences between value table and prior value table

        returns:
            - max change value
    """
    def find_max_vchange(self, vtable_diff):
        # max_(s in S) abs ( V_t (s) - V_t-1(s) )
        rows = self.env.nrows()
        cols = self.env.ncols()
        vopts = self.vl_opts
        max_r = max_c = max_vr = max_vc = 0
        max_change = -1

        for r in range(rows):
            for c in range(cols):
                for vr in vopts:
                    for vc in vopts:
                        current = vtable_diff[r, c, vr, vc]
                        if (current > max_change):
                                max_r = r
                                max_c = c
                                max_vr = vr
                                max_vc = vc
                                max_change = current

        return max_change