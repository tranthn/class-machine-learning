#!/usr/bin/env python3
import numpy as np
import random
from termcolor import colored, cprint

"""
    – S – This square is on the starting line.
    – F – This square is on the finish line.
    – . – This square is open racetrack.
    – # – This square is off the racetrack (i.e., a wall).
"""

class TrackSimulator():
    def __init__(self, track = None, start_pos = (0, 0), start_velocity = (0, 0)):
        self.track = track

        # position = r, c - r = row, c = column to match dataframe/matrix representation
        self.position = start_pos
        self.velocity = start_velocity

        # array of coordinates to indicate path taken (for printing)
        self.path = []

    # helper that returns coordinates of a given feature in track (S, F, ./track, #/wall)
    def _find_coordinate(self, char = ''):
        for coords, value in np.ndenumerate(self.track):
            if (value == char):
                return coords
        return (-1, -1)

    def pretty_print(self):
        one_spacer = ' '
        two_spacer = '  '
        three_spacer = '   '
        r = 0

        # print column headers, with spacers to handle single vs. double-digit numbers
        print('\t   ', end = '')
        for i in range(0, self.track.shape[1]):
            spacer = one_spacer if i > 9 else two_spacer
            cprint('{0}{1}'.format(i, spacer), 'red', end = '')

        for coords, value in np.ndenumerate(self.track):
            # print row headers
            if (coords == (0, 0) or r < coords[0]):
                spacer = one_spacer if coords[0] > 9 else two_spacer
                print()
                cprint('\t{0}{1}'.format(coords[0], spacer), 'red', end = '')

            spacer = two_spacer 

            # if we're on current position or on path, print with color for visual indication
            if (self.position == coords or (coords[0], coords[1]) in self.path):
                cprint(value + spacer, 'green', end = '')
            else:
                print(value + spacer, end = '')

            r = coords[0]
        
        print('\n')
    

####################################################################################
    def initialize_track(self):
        self.position = self._find_coordinate('S')

    def boundary_check(self, position):
        r = position[0]
        c = position[1]
        rows = self.track.shape[0]
        cols = self.track.shape[1]

        print('new r,c:', r, c)

        # check for out of bounds coordinates
        if (r < 0 or c < 0 or r >= rows or c >= cols):
            cprint('out of bounds', 'magenta')
            return False

        # reverse y, x since y indicates row position, while x indicates column position
        elif (self.track[r, c] == '#'):
            cprint('hit a wall', 'magenta')
            return False
        else:
            return True

    def move(self):
        v = self.velocity
        p1 = self.position[0]
        p2 = self.position[1]

        position = (v[0] + p1, v[1] + p2)
        if (self.boundary_check(position)):
            self.position = position
            self.path.append(self.position)
        else:
            cprint('offtrack', 'magenta')
            # TODO two handling strategies
            # 1 - move to x,y closest to crash point
            # 2 - move to start

        print()

    def accelerate(self, rise, run):
        percent = round(random.random() * 100, 0)
        if (percent > 20):  
            self.velocity[0] += rise
            self.velocity[1] += run
            cprint('accelerate success', 'green')
        else:
            cprint('accelerate failed', 'red')

    ## helper that runs through with predetermined test path to finish line
    ## track is initialized on a start position already
    def test_run(self):
        self.velocity = [0, 1]
        for i in range(10):
            self.accelerate(0, 1) # rise, run
            self.move()