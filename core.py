from agent.DQNAgent import DQNAgent
from environment.TicTacToe import TicTacToe

def main():
    env = TicTacToe()

    state_size = 19683
    action_size = 9

    log_actions_freq = 400

    agent = DQNAgent(state_size, action_size)
    target_mode = False

    for e in range(agent.episodes):
        agent.reset()
        env.restart()
        state = env.getState()
        episode_len = 0

        #one episode
        while True:
            env.render()

            action = agent.act(state)

            reward = env.step(action)
            reward = float(reward)
            next_state = env.getState()
            done = env.terminal

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done:
                if agent.loger_mode:
                    with open(agent.log_path, "a") as log_file:
                        log_file.write("episode: {}/{}, episodelen: {}, e: {:.2}\n"
                        .format(e, agent.episodes, episode_len, agent.epsilon))
                else:
                    print("episode: {}/{}, episodelen: {}, e: {:.2}"
                        .format(e, agent.episodes, episode_len, agent.epsilon))

                if agent.epsilon > agent.epsilon_min:
                    agent.epsilon *= agent.epsilon_decay
                if e % agent.save_model_freq == 0:
                    agent.model.save(agent.model_path+str(e/agent.save_model_freq))

                agent.update()

                break

            env.invert_board()
            episode_len += 1


if __name__ == '__main__':
    main()