"""Reinforcement learning for Tic Tac Toe as in
https://webdocs.cs.ualberta.ca/~sutton/book/ebook/node10.html"""

import sys
import numpy as np

BOARD_SIZE = (3, 3)

class Board(object):
    EMPTY = 0
    CROSS = 1
    CIRCL = 2

    def __init__(self):
        self.state = np.zeros(BOARD_SIZE, dtype='I')

    def set_cross(self, coord):
        """Set a cross at tuple coord"""
        pass

    def set_circl(self, coord):
        """Set a circle at tuple coord"""
        pass

    def is_done(self):
        return True

    def who_won(self):
        return self.CIRCL

    def get_state_string(self):
        return self.state.tostring()

    def convert_to_icon(self, value):
        if value == self.EMPTY:
            return ' '
        elif value == self.CROSS:
            return 'X'
        elif value == self.CIRCL:
            return 'O'

    def print_board(self):
        for y in range(BOARD_SIZE[1]):
            for x in range(BOARD_SIZE[0]):
                sys.stdout.write(self.convert_to_icon(self.state[x, y]))
                if x < BOARD_SIZE[0]-1:
                    sys.stdout.write(" | ")
            if y < BOARD_SIZE[1]-1:
                sys.stdout.write("\n---------")
            sys.stdout.write("\n")

class Agent(object):
    STARTING_VALUE = 0.5
    def __init__(self, filename):
        if not filename:
            print "Starting with new value map"
        self.state = np.zeros(BOARD_SIZE, dtype='I').tostring()
        self.values = {self.state: np.ones(BOARD_SIZE)*self.STARTING_VALUE}

    def set_state(self, state):
        self.state = state
        if not state in self.values:
            self.values[state] = np.ones(BOARD_SIZE)*self.STARTING_VALUE

    def get_action(self):
        """Choose the action with the highest probability"""
        action = np.argmax(self.values)
        return action

def ask_user():
    """Get coordinates from user via stdin return a tuple"""

def main():
    filename = None
    try:
        filename = sys.argv[1]
    except IndexError:
        pass
    agt = Agent(filename)
    brd = Board()
    while not brd.is_done():
        brd.set_cross(ask_user())
        agt.set_state(brd.get_state_string())
        brd.set_circl(agt.get_action())
        brd.print_board()

if __name__ == '__main__':
    main()
