import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = 2000
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_rate = 0.0001

        self.memory = deque(maxlen=9)


        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 50
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = True

        self.act_log_path = "logs/act_log.txt"

        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(32, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def constraint_predict(self, state):
        act_values = self.model.predict(state)

        max_index = 0
        max_value = 0

        for i in range(9):
            if state[i] == 0:
                max_value = act_values[i]
                break

        for i in range(9):
            if state[i] == 0:
                if max_value > act_values[i]:
                    max_value = act_values[i]
                    max_index = i

        return max_index

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            indexes = []
            for i in range(9):
                if state[i] == 0: indexes.append(i)


            rand_index = random.randrange(len(indexes))
            return indexes[rand_index]
        else:
            return self.constraint_predict(state)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def reset(self):
        self.memory.clear()

    def update(self):
        reward = self.memory[len(self.memory)-1][2]

        for i in reversed(range(len(self.memory)-1)):
            self.memory[i] = (self.memory[i][0], self.memory[i][1], reward, self.memory[i][3], self.memory[i][4])
            reward *= -1.0
            reward *= self.gamma

        for i in reversed(range(len(self.memory))):
            state, action, reward, state_next, done = self.memory[i]
            target = reward

            target = (reward + state_next[self.constraint_predict(state_next)])

            target_f = state[self.constraint_predict(state)]
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)