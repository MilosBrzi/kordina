from agent.Player import Player

class HumanPlayer(Player):
    '''
    This class implements Player abstract class
    such that it allows human user to make
    a desired action using console input
    '''

    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)

    def act(self, state):
        action = input('Enter action: ')
        action = int(action)
        action -= 1
        if action <= 2:
            action+=6
        elif action >=6:
            action-=6

        return int(action)