#!/usr/bin/env python3
import numpy as np
from termcolor import colored, cprint

class Learner():

    def __init__(self, learning_algorithm = 'q'):
        self.algo = learning_algorithm
        self.track = track

    # updates based on the immediately experienced / preceding Q-value
    def sarsa(self):
        return None

################################################################################

class QLearner():
    def __init__(self, track = None):
        # TrackSimulator object
        self.track = track

    def pretty_print(self):
        one_spacer = ' '
        two_spacer = '  '
        three_spacer = '   '
        r = 0

        # print column headers, with spacers to handle single vs. double-digit numbers
        print('\t   ', end = '')
        for i in range(0, self.state_table.shape[1]):
            spacer = one_spacer if i > 9 else two_spacer
            cprint('{0}{1}'.format(i, spacer), 'red', end = '')

        for coords, value in np.ndenumerate(self.state_table):
            # print row headers
            if (coords == (0, 0) or r < coords[0]):
                spacer = one_spacer if coords[0] > 9 else two_spacer
                print()
                cprint('\t{0}{1}'.format(coords[0], spacer), 'red', end = '')

            spacer = one_spacer 
            print(value, spacer, end = '')
            r = coords[0]
        
        print('\n')

    # model free, we do not know state transition probabilities or rewards
    # goal is to learn value-function Q
    # uses epsilon to determine how greedy (= 0) or optimal (> 0)
    #
    # agent is run in environment as trial, state-action pairs collected
    #   - Q_t(s, a): estimate value for (s, a) pair
    #   - alpha_t(s, a) in [0, 1]: learning rate at given time *t*
    #   - gamma in [0, 1]: discount factor
    #   - s': successor state
    #
    # updates based on what would have been optimal according to current estimate of our Q function
    # selects next option using softmax to pick optimal choice
    def initialize_table(self):
        dims = self.track.shape
        self.state_table = np.full(dims, 0, dtype=int)
        self.pretty_print()