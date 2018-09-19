import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam


class Player:
    #mode == 1 Training
    #mode == 0 Testing
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

    def act(self, state):
        action = input('Enter action: ')
        return int(action)