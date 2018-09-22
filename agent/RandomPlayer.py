import random
from environment.TicTacToe import TicTacToe
from agent.Player import Player

class RandomPlayer(Player):
    '''
    This class implements Player abstract class
    in such way that it creates a Player whose
    can return the move by choosing random action
    from valid action list based on current state
    '''

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

    def act(self, state):
        indexes = TicTacToe.valid_moves(state)
        rand_index = random.randrange(len(indexes))
        return indexes[rand_index]