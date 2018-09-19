import numpy as np


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype="int")
        self.terminal = False
        self.to_move = 1

    def restart(self):
        self.board = np.zeros((3, 3), dtype="int")
        self.terminal = False
        self.to_move = 2

    def is_valid(self, action):
        if self.getState()[action] != 0:
            return False
        else:
            return True

    @staticmethod
    def valid_moves(state):
        val_moves = []
        for i in range(9):
            if state[i] == 0: val_moves.append(i)
        return val_moves

    @staticmethod
    def invalid_moves(state):
        inval_moves = []
        for i in range(9):
            if state[i] != 0: inval_moves.append(i)
        return inval_moves

    def invert_player(self):
        if self.to_move == 1: self.to_move=2
        else : self.to_move = 1

    def invert_board(self):
        for row in range(3):
            for col in range(3):
                if (self.board[row][col] == 1):
                    self.board[row][col] = -1
                elif (self.board[row][col] == -1):
                    self.board[row][col] = 1
        self.invert_player()

    def step(self, action):
        """
        PARAMS: a valid action (int 0 to 8)
        RETURN: reward (-1,0,1)
        self.board    is updated in the process
        self.terminal is updated in the process
        """

        # insert
        row_index = int(np.floor(action / 3))
        col_index = action % 3
        self.board[row_index][col_index] = 1

        # undecided
        terminal = 1

        # to check for 3 in a row horizontal
        for row in range(3):
            for col in range(3):
                if (self.board[row][col] != 1):
                    terminal = 0
            if (terminal == 1):
                self.terminal = True
                return +1
            else:
                terminal = 1

        # to check for 3 in a row vertical
        for col in range(3):
            for row in range(3):
                if (self.board[row][col] != 1):
                    terminal = 0
            if (terminal == 1):
                self.terminal = True
                return +1
            else:
                terminal = 1

        # diagonal top-left to bottom-right
        for diag in range(3):
            if (self.board[diag][diag] != 1):
                terminal = 0
        if (terminal == 1):
            self.terminal = True
            return +1
        else:
            terminal = 1

        # diagonal bottom-left to top-right
        for diag in range(3):
            if (self.board[2 - diag][diag] != 1):
                terminal = 0
        if (terminal == 1):
            self.terminal = True
            return +1
        else:
            terminal = 1

        # checks if board is filled completely
        for row in range(3):
            for col in range(3):
                if (self.board[row][col] == 0):
                    terminal = 0
                    break
        if terminal == 1:
            self.terminal = True

        self.invert_player()

        return 0

    def render(self):
        """
        print to screen the full board nicely
        """

        for i in range(3):
            print('\n|', end="")
            for j in range(3):
                if self.board[i][j] == 1:
                    print(' X |', end="")
                elif self.board[i][j] == 0:
                    print('   |', end="")
                else:
                    print(' O |', end="")
        print('\n')

    def getState(self):
        state = []
        for i in range(3):
            for j in range(3):
                state.append(self.board[i][j])

        return state