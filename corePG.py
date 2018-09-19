from agent.DQNAgent import DQNAgent
from agent.PolicyAgent import PolicyAgent
from environment.TicTacToe import TicTacToe
from random import randint
from agent.RandomPlayer import RandomPlayer
import random

def play_quick_game(agent, env):

    player = RandomPlayer(9,9, env)
    result = 0
    num_of_agent_lost_games = 0
    for i in range(10):
        env.restart()
        agent.reset()

        state = env.getState()
        episode_len = 0

        r = random.uniform(0, 1)
        if r > 0.5: r = 0
        else: r = 1
        # one episode
        while True:
            if episode_len % 2 == r:
                action, _ = agent.act(state, clean=True)
            else:
                action = player.act(state)

            reward = env.step(action)

            state = env.getState()
            done = env.terminal

            if done:
                whowon = 0
                if episode_len % 2 == r:
                    whowon = reward
                else:
                    whowon = reward * -1
                result += whowon
                if whowon == -1:
                    num_of_agent_lost_games += 1
                break

            episode_len += 1

            env.invert_board()

    return num_of_agent_lost_games


def main():
    env = TicTacToe()

    state_size = 19683
    state_dim = 9
    action_dim = 9

    log_actions_freq = 400

    agent = PolicyAgent(1, state_dim, action_dim)

    for e in range(agent.episodes):
        agent.reset()
        env.restart()
        state = env.getState()
        episode_len = 0
        actions_played = []
        #one episode
        while True:
            #env.render()

            action, _ = agent.act(state)
            #print(norm_act_prob)

            actions_played.append(action)

            reward = env.step(action)
            reward = float(reward)
            env.invert_board()
            next_state = env.getState()
            done = env.terminal

            agent.remember_move(state, action, reward)

            state = next_state

            if done:
                whowon = 0
                if episode_len % 2 == 0:
                    whowon = reward
                else:
                    whowon = reward * -1
                '''
                if agent.loger_mode:
                    with open(agent.log_path, "a") as log_file:
                        log_file.write("episode: {}/{}, episodelen: {}, whowon: {},\n"
                                       "actions player: {}"
                        .format(e, agent.episodes, episode_len, whowon, actions_played))
                else:
                    print("episode: {}/{}, episodelen: {}, whowon: {}, \n"
                                       "actions player: {}"
                        .format(e, agent.episodes, episode_len, whowon, actions_played))
                '''

                if e % agent.save_model_freq == 0:
                    agent.model.save(agent.model_path+str(e/agent.save_model_freq))

                agent.remember_game()

                if e % agent.batch_freq == 0 and e > 4000:
                    agent.update_batch()

                if e % 100 == 0:
                    result = play_quick_game(agent, env)
                    with open(agent.log_path, "a") as log_file:
                        log_file.write("" + str(result) + "\n")
                break

            episode_len += 1


if __name__ == '__main__':
    main()