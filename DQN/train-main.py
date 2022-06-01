import numpy as np
# from utils import *  # 开始前的import为了能够让配置和一些预处理的函数  在主程序开始前就生效
from agent import *
from config import *
from rl4uc.environment import *


def train(env, agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    rewards_log = []
    average_log = []
    eps = eps_init

    for i in range(1, 1 + num_episode):

        episodic_reward = 0
        done = False
        state = env.reset()
        t = 0  # 这里的t 可以看成one episode 里面的 timestep

        while not done and t < max_t:

            action, processed_obs = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            next_state_processs = agent.process_observation(next_state)
            agent.memory.store(processed_obs, action, reward, next_state_processs, done)
            # state = next_state.copy()
            state = next_state

            if agent.memory.is_full():
                agent.memory.reset_buffer()

            t += 1
            print("timesteps：{}".format(t))
            print("reward：{}".format(reward))

            if t % 4 == 0 and agent.memory.store_num >= agent.bs:
                agent.update(agent.memory)
                agent.soft_update(agent.tau)

            episodic_reward += reward  # 记录一个episode的奖励

        rewards_log.append(episodic_reward)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]), end='')
        if i % 50 == 0:
            print()

        eps = max(eps * eps_decay, eps_min)

    return rewards_log, average_log


'根据那个写test 和train 模式的切换'

if __name__ == '__main__':
    env = make_env_from_json('5gen')

    agent = Agent(env, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    rewards_log, average_log = train(env, agent, NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    np.save('Data/{}_rewards2.npy'.format(ENV_NAME), rewards_log)
    np.save('Data/{}_average.npy'.format(ENV_NAME), average_log)

    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), 'Data/{}_weights2.pth'.format(ENV_NAME))
