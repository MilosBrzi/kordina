import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from environment.TicTacToe import TicTacToe

class RandomPlayer:
    #mode == 1 Training
    #mode == 0 Testing
    def __init__(self, state_dim, action_dim, game):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.game = game

    def act(self, state):
        indexes = self.game.valid_moves(state)
        rand_index = random.randrange(len(indexes))
        return indexes[rand_index]