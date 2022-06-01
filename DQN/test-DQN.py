from rl4uc.environment import *
import matplotlib.pyplot as plt
from config import *
import torch
from agent import Agent


def test():
    env = make_env(mode='test', profiles_df='test_data_5gen.csv')
    load = env.profiles_df.demand
    print(env.gen_info)
    print(env.episode_length)
    agent = Agent(env, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    agent.load_state_dict()

    obs = env.reset()
    actions = []
    epoch_timesteps = []
    epoch_rewards = []
    dispatch = []
    timesteps = 0
    done = False
    while not done:
        action, processed_obs = agent.act(obs)
        next_obs, reward, done = env.step(action)
        next_obs_processed = agent.process_observation(next_obs)
        obs = next_obs
        timesteps += 1
        print('timestep{} action is {}'.format(timesteps, action))
        # print("timesteps：{}".format(timesteps))
        print("reward：{}".format(reward))
        print("timestep{} dispatch{}".format(timesteps, env.disp))
        epoch_rewards.append(reward)
        epoch_timesteps.append(timesteps)
        dispatch.append(env.disp)
        actions.append(action)
        if done:
            print('Test in timestep{} is done'.format(timesteps))
            break

    return epoch_rewards, epoch_timesteps, dispatch, agent, load


def visualize_results(epoch_rewards, epoch_timesteps, dispatch, load):
    plt.plot(epoch_timesteps, epoch_rewards)
    plt.show()
    dispatch_process = np.array(dispatch)
    dispatch_all = np.sum(dispatch_process, axis=1)
    plt.bar(epoch_timesteps, dispatch_process[:, 0], alpha=0.5, label='G1')
    plt.bar(epoch_timesteps, dispatch_process[:, 1])
    plt.bar(epoch_timesteps, dispatch_process[:, 2])
    plt.bar(epoch_timesteps, dispatch_process[:, 3])
    plt.bar(epoch_timesteps, dispatch_process[:, 4])
    plt.plot(epoch_timesteps, load)

    plt.legend()
    plt.show()
epoch_rewards, epoch_timesteps, dispatch, agent, load = test()
visualize_results()