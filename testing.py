from agent.DQNAgent import DQNAgent
from agent.PGAgent import PGAgent
from agent.HumanPlayer import HumanPlayer
from environment.TicTacToe import TicTacToe
import random

def main():
    env = TicTacToe()

    state_dim = 9
    action_dim = 9

    agent = DQNAgent(0, state_dim, action_dim)
    player = HumanPlayer(state_dim, action_dim)

    for e in range(agent.episodes):
        agent.reset()
        env.restart()
        state = env.getState()
        episode_len = 0

        r = random.uniform(0, 1)
        if r > 0.5:
            r = 0
        else:
            r = 1
        # one episode
        while True:
            env.render()

            if episode_len % 2 == r:
                action = agent.act(state, clean=True)
            else:
                action = player.act(state)

            env.step(action)

            next_state = env.getState()
            done = env.terminal

            state = next_state

            if done:
                env.render()
                break

            episode_len += 1
            env.invert_board()

if __name__ == '__main__':
    main()