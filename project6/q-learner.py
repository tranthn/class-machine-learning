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

""" PSEUDOCODE
short hands
--------------------------------------------
    gm = discount factor
    epsilon = exploration factor, 0 = greedy, higher value = better results, slower performance
    alpha = learning rate
    R(s, a) = reward of state, action combo
    
Q_t+1(s, a) = (1 - alpha_t (s, a)) * Q_t(s, a) + alpha_t (s, a) [ reward(s, a) + gamma * max (Q_t(s', a')) ]
"""

class QLearner():
    def __init__(self, env = None, vl_opts = [-1, 0, 1], 
                        actions = [(0,0), (0,1), (1,0)],
                        alpha = 0.5, gamma = 1.0, epsilon = 0.1):
        
        # TrackSimulator object
        self.env = env
        self.vl_opts = vl_opts
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

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
        self.initialize_q_table()

    def initialize_q_table(self):
        # this table needs to hold Q-values
        # i.e. all positionals x all velocity x action combos
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

    def value_iteration(self, iterations = 5):
        self.initialize()
        rows = self.env.nrows()
        cols = self.env.ncols()
        vopts = self.vl_opts
        track = self.env.track
        threshold = 0.8
        epsilon = self.epsilon
        gamma = self.gamma
        lr = self.alpha
        max_vchange = 100

        # q-value for finish states
        for r in range(rows):
            for c in range(cols):
                if (track[r, c]) == 'F':
                    for vc in vopts:
                        for vr in vopts:
                            for aidx, action in enumerate(self.actions):
                                self.qtable[r, c, vr, vc, aidx] = 0

        for i in range(iterations):
            print('\nq-learning iteration, #', i)

            r = np.random(range(rows))
            c = np.random(range(cols))
            vr = np.random(range(vopts))
            vc = np.random(range(vopts))

            for r in range(10):
                # set current running reward, 0 if on finish cell
                if (track[r, c] == 'F' or track[r, c] == '#'):
                    break
                    
                # pick next action
                act_maxq_idx = np.argmax(self.qtable[r, c, vr, vc])
                maxq = self.qtable[r, c, vr, vc, act_maxq_idx]
                next_action = self.actions(act_maxq_idx)

                # set current position/coordinates based on loop values
                self.env.position = pos1 = (r, c)
                self.env.velocity = vel1 = (vr, vc)

                # take current action on track simulator
                # accelerate(): internally handles chance of acceleration failure and adjusts velocity accordingly
                # move(): handles updating position and restarting if crashed
                next_vel = self.env.accelerate(action[0], action[1])
                next_pos = self.env.move()
                self.env.finalize_move()
                reward = -1

                # take transition probabilities into account
                # val_new = vtable_prior[next_pos[0], next_pos[1], next_vel[0], next_vel[1]]
                # val_no_acc = vtable_prior[pos_no_acc[0], pos_no_acc[1], vel1[0], vel1[1]]
                future_reward = reward + gamma * self.qtable[next_pos[0], next_pos[1], next_vel[0], next_vel[1], act_maxq_idx]
                qvalue = lr * self.qtable[r, c, vr, vc, act_maxq_idx]

        # store final optimal action for a given state combo
        policy = self.build_policy(rows, cols, vopts)

        return policy

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