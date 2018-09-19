from agent.DQNAgent import DQNAgent
from agent.PolicyAgent import PolicyAgent
from agent.Player import Player
from environment.TicTacToe import TicTacToe

def main():
    env = TicTacToe()

    state_size = 19683
    state_dim = 9
    action_dim = 9

    log_actions_freq = 400

    agent = PolicyAgent(0, state_dim, action_dim)
    player = Player(state_dim, action_dim)

    for e in range(agent.episodes):
        agent.reset()
        env.restart()
        state = env.getState()
        episode_len = 0

        #one episode
        while True:
            env.render()

            if episode_len % 2 == 0:
                action, _= agent.act(state, clean = True)
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