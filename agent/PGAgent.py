import random
from environment.TicTacToe import TicTacToe
from collections import deque
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from agent.Agent import Agent

class PGAgent(Agent):
    #mode == 1 Training
    #mode == 0 Testing
    def __init__(self, mode, state_dim, action_dim):
        super().__init__(state_dim, action_dim, mode)
        self.episodes = 100001
        self.gamma = 0.5  # discount rate

        self.learning_rate = 0.00002

        self.one_game_memory = deque(maxlen=9)

        self.batching_memory = deque(maxlen=50000)
        self.batch_size = 1024
        self.batch_freq = 64

        self.model_path = "models/PG_model_e_"
        self.save_model_freq = 5000
        self.log_path = "logs/pg_log.txt"
        self.loger_mode = False

        self.model = self._build_model()
        if mode == 0:
            self.trained_model_path = self.model_path+"Alpha"
            self.load(self.trained_model_path)

    def _build_model(self):
        # Neural Net for PG learning Model
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_dim, activation='relu', kernel_initializer ='he_uniform'))
        model.add(Dense(512, activation='relu', kernel_initializer ='he_uniform'))
        model.add(Dense(512, activation='relu', kernel_initializer ='he_uniform'))
        model.add(Dense(self.action_dim, activation='softmax'))
        model.compile(optimizer=RMSprop(lr=self.learning_rate, decay=1e-11),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])
        model.summary()

        return model

    def best_action(self, state, act_prob, clean = False):
        possible_actions = TicTacToe.valid_moves(state)

        if clean == False:
            act_index = random.choice(possible_actions)
            return act_index
        else:
            probability_of_actions = []
            for i in possible_actions:
                probability_of_actions.append(act_prob[i])

            #probability_of_actions = probability_of_actions / np.sum(probability_of_actions)
            ind =  probability_of_actions.index(max(probability_of_actions))
            return possible_actions[ind]
            #act_index = np.random.choice(possible_actions, 1, p=probability_of_actions)
            #return act_index[0]

    def act(self, state, clean=False):
        act_prob = self.model.predict(np.expand_dims(np.asarray(state), 0), batch_size=1)
        act_prob = act_prob[0]

        act_index = self.best_action(state, act_prob, clean)
        return act_index

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def remember_move(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros([self.action_dim])
        one_hot_action[action] = 1

        self.one_game_memory.append((state, one_hot_action, reward))

    def remember_game(self):
        self.calculate_reward()

        states = [gm[0] for gm in self.one_game_memory]

        one_hot_action = [gm[1] for gm in self.one_game_memory]
        rewards = [gm[2] for gm in self.one_game_memory]
        rewards = np.vstack(rewards)

        targets = one_hot_action * rewards

        for i in range(len(self.one_game_memory)):
            self.batching_memory.append((states[i], targets[i]))

    def update_batch(self):
        minibatch = random.sample(self.batching_memory, self.batch_size)

        states = [gm[0] for gm in minibatch]
        states = np.vstack(states)
        targets = [gm[1] for gm in minibatch]
        targets = np.vstack(targets)

        loss = self.model.train_on_batch(states, targets)[0]
        print(loss)

    def reset(self):
        self.one_game_memory.clear()

    def calculate_reward(self):
        disc_reward = self.one_game_memory[len(self.one_game_memory) - 1][2]
        if disc_reward == 1.0:
            flip = True
            for i in reversed(range(len(self.one_game_memory))):
                reward = -disc_reward if flip == False else disc_reward

                self.one_game_memory[i] = (self.one_game_memory[i][0], self.one_game_memory[i][1], 1000 * reward)
                flip = not flip
                if flip:
                    disc_reward *= self.gamma
        else:
            disc_reward = 0.1
            flip = True
            for i in reversed(range(len(self.one_game_memory))):
                reward = disc_reward
                self.one_game_memory[i] = (self.one_game_memory[i][0], self.one_game_memory[i][1], 1000 * reward)
                flip = not flip
                if flip:
                    disc_reward *= self.gamma