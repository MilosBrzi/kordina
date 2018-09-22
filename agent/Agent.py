import random
from environment.TicTacToe import TicTacToe
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from agent.Player import Player

class Agent(Player):
    #mode == 1 Training
    #mode == 0 Testing
    def __init__(self, state_dim, action_dim, mode):
        super().__init__(state_dim, action_dim)
        self.mode = mode

    def _build_model(self):
        pass

    def best_action(self, state, act_values):
        max_index = 0
        max_value = 0

        for i in range(self.action_dim):
            if state[i] == 0:
                max_value = act_values[i]
                break

        for i in range(self.action_dim):
            if state[i] == 0:
                if act_values[i] >= max_value:
                    max_value = act_values[i]
                    max_index = i

        return max_index, max_value

    def act(self, state, clean=False):
        pass

    def load(self, name):
        pass

    def save(self, name):
        pass

    def remember_move(self, state, action, reward, next_state, done):
        pass

    def remember_game(self):
        pass

    def reset(self):
        pass

    def update_batch(self):
        pass

    def calculate_reward(self):
        pass