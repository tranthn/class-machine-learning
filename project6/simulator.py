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

MIN_VELOCITY = -5
MAX_VELOCITY = 5

class TrackSimulator():
    def __init__(self, track = None):
        self.track = track

        # position = r, c 
        # r = row, c = column to match dataframe/matrix representation
        self.position = (0,0)
        self.velocity = (0,0)

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
                cprint('X' + spacer, 'green', end = '')
            else:
                print(value + spacer, end = '')

            r = coords[0]
        
        print('\n')
    
####################################################################################
####################################################################################

    # sets initial start position at first S cell found
    def initialize_track(self):
        self.position = self._find_coordinate('S')
        self.start_pos = self.position

    # helper that returns all open positions within racetrack
    def get_all_track_points(self):
        open_track = []
        for coords, value in np.ndenumerate(self.track):
            if (value == '.'):
                open_track.append(coords)

        return open_track

    # checks if given position hits a wall or goes out of bounds
    def boundary_check(self, position):
        r = position[0]
        c = position[1]
        rows = self.track.shape[0]
        cols = self.track.shape[1]

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

    """
        two handling strategies
            1 - move to x,y closest to crash point
            2 - move to start
    """
    def get_restart_position(self, crash_site, goto_start = False):
        if not goto_start:
            # TODO implement actual logic that grabs closest open point, TBD if we can clip through wall
            # track_pts = self.get_all_track_points()
            # print(track_pts)
            return self.position
        else:
            return self.start_pos

    # gives next position based off current position and velocity
    def get_next_position(self):
        v = self.velocity
        p1 = self.position[0]
        p2 = self.position[1]
        position = (self.velocity[0] + self.position[0],
                    self.velocity[1] + self.position[1])

        return position

    """
        Attempts to accelerate, i.e. adjust the velocity.
        Has 80% of succeeding and adjusts velocity within [-5, 5] limit

        arguments:
            - rise: the y-value or which row we adjust to within the 2d track array
            - run: the x-value or which column we adjust to within the 2d track array
    """
    def accelerate(self, rise, run):
        percent = round(random.random() * 100, 0)
        if (percent > 20):  
            vr = self.velocity[0] + rise
            vc = self.velocity[1] + run

            # ensure velocity doesn't go below min velocity or above max velocity
            cprint('accelerate success', 'green')
            print('v1', self.velocity)
            self.velocity[0] = max(vr, MIN_VELOCITY) if (vr < 0) else min(vr, MAX_VELOCITY)
            self.velocity[1] = max(vc, MIN_VELOCITY) if (vc < 0) else min(vc, MAX_VELOCITY)
            print('v2', self.velocity)
        else:
            cprint('accelerate failed', 'red')
            print('v1', self.velocity)

    # method tries to adjust position, checking boundaries first
    # if new position goes off track, we will restart position
    def move(self):
        cprint('move', 'green')
        print('pos1', self.position)
        position = self.get_next_position()
        print('pos2', position)
        if (self.boundary_check(position)):
            self.position = position
            self.path.append(self.position)
        else:
            cprint('restart', 'yellow')
            self.velocity = [0,0]
            self.position = self.get_restart_position(position)

        print()

    ## helper that runs through with predetermined test path to finish line
    ## track is initialized on a start position already
    def test_run(self):
        self.velocity = [0, 1]
        for i in range(10):
            self.accelerate(0, 1) # rise, run
            self.move()