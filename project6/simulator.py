#!/usr/bin/env python3
import sys
import numpy as np
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
        r = 0

        # print column headers, with spacers to handle single vs. double-digit numbers
        print('\t  ', end = '')
        for i in range(0, self.track.shape[1]):
            spacer = ' '
            if (i < 10):
                spacer = '  '
            
            print('{0}{1}'.format(i, spacer), end = '')

        for coords, value in np.ndenumerate(self.track):

            # print row headers
            if (coords == (0, 0) or r < coords[0]):
                print()
                print('\t{0} '.format(coords[0]), end = '')

            # if we're on current position or on path, print with color for visual indication
            spacer = '  ' 
            if (self.position == coords or (coords[0], coords[1]) in self.path):
                cprint(value + spacer, 'green', end = '')
            else:
                print(value + spacer, end = '')

            r = coords[0]
        
        print('\n')
    

####################################################################################
    def initialize_track(self):
        self.position = self._find_coordinate('S')

    def move(self, x, y):
        self.position[0] += x
        self.position[1] += y

    def accelerate(self, x, y):
        self.velocity[0] += x
        self.velocity[1] += y

    ## helper that runs through with predetermined test path
    def test_run(self):
        return None