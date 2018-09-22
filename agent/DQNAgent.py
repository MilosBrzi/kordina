import random
import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from environment.TicTacToe import TicTacToe
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import Adam
from agent.Agent import Agent

class DQNAgent(Agent):
    def __init__(self, mode, state_dim, action_dim):
        super().__init__(state_dim, action_dim, mode)
        self.episodes = 500001
        self.gamma = 0.3  # discount rate
        if mode == 1:
            self.epsilon = 1.00  # exploration rate
        else: self.epsilon = 0

        self.epsilon_min = 0.8
        self.epsilon_decay = 0.99998
        self.learning_rate = 0.00001

        self.one_game_memory = deque(maxlen=9)

        self.batching_memory = deque(maxlen=50000)
        self.batch_size = 512
        self.batch_freq = 16

        self.model_path = "models/DQN_model_e_"
        self.save_model_freq = 5000
        self.log_path = "logs/dqn_log.txt"
        self.loger_mode = False

        self.act_log_path = "logs/act_log.txt"

        self.model = self._build_model()
        if mode == 0:
            self.trained_model_path = self.model_path+"10.0"
            self.load(self.trained_model_path)

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='relu', batch_size=self.batch_size))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        model.summary()

        return model

    def best_action_prediction(self, state):
        act_values = self.model.predict(np.expand_dims(np.asarray(state), 0), batch_size=1)
        act_values = act_values[0]
        return self.best_action(state, act_values)

    def act(self, state, clean=False):
        if np.random.rand() <= self.epsilon and not clean:
            indexes = []
            for i in range(9):
                if state[i] == 0: indexes.append(i)

            rand_index = random.randrange(len(indexes))
            return indexes[rand_index]

        else:
            act_index, act_value = self.best_action_prediction(state)
            return act_index

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember_move(self, state, action, reward, next_state, done):
        self.one_game_memory.append((state, action, reward, next_state, done))

    def remember_game(self):
        self.calculate_reward()
        for i in range(len(self.one_game_memory)):
            self.batching_memory.append(self.one_game_memory[i])

    def reset(self):
        self.one_game_memory.clear()

    def update_batch(self):
        minibatch = random.sample(self.batching_memory, self.batch_size)

        state_mb = [mb[0] for mb in minibatch]
        state_mb = np.asarray(state_mb)
        actions = [mb[1] for mb in minibatch]
        rewards = [mb[2] for mb in minibatch]
        state_next_mb = [mb[3] for mb in minibatch]
        state_next_mb = np.asarray(state_next_mb)
        dones = [mb[4] for mb in minibatch]

        targets = rewards

        act_values_mb = self.model.predict(state_next_mb, batch_size=self.batch_size)

        act_values = []
        for i in range(self.batch_size):
            state = state_next_mb[i]
            act_value = act_values_mb[i]
            max_index, max_value = self.best_action(state, act_value)
            act_values.append(max_value)

        for i in range(self.batch_size):
            if not dones[i]:
                targets[i] = rewards[i] + act_values[i]

        targets_f = self.model.predict(state_mb)

        for i in range(self.batch_size):
            targets_f[i][actions[i]] = targets[i]
            for inv_index in TicTacToe.invalid_moves(state_mb[i]):
                targets_f[i][inv_index] = -1

        self.model.fit(state_mb, targets_f, batch_size=self.batch_size, epochs=1, verbose=0)

    def calculate_reward(self):
        reward = self.one_game_memory[len(self.one_game_memory) - 1][2]

        flip = True
        for i in reversed(range(len(self.one_game_memory))):
            self.one_game_memory[i] = (self.one_game_memory[i][0], self.one_game_memory[i][1], reward, self.one_game_memory[i][3], self.one_game_memory[i][4])
            reward *= -1.0
            flip = not flip
            if flip:
                reward *= self.gamma