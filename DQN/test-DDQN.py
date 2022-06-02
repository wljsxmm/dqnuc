import matplotlib.pyplot as plt
import numpy as np
from agent import DDQN_Agent
from config import *
# from rl4uc.environment import *  # 不要这样导入 会导致 将environment的函数先执行一遍 不过问题不大
from rl4uc.environment import make_env


def test():
    env = make_env(mode='test', profiles_df='test_data_5gen.csv')
    # day, day_profile = env.sample_day()
    # load = day_profile.demand.values
    # wind = day_profile.wind.values

    # load = env.episode_forecast  "打印测试文件里面的全部的数据"
    # wind = env.episode_wind_forecast

    obs = env.reset()
    load = env.episode_forecast
    wind = env.episode_wind_forecast

    parameter_path = 'Data/DDQN/{}_weights_test1.pth'.format(ENV_NAME)
    # parameter_path = 'Data/{}_weights2.pth'.format(ENV_NAME)
    print(env.gen_info)
    print('Test episode length is {}'.format(env.episode_length_test))
    agent = DDQN_Agent(env.num_gen, obs, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    agent.load_state_dict(parameter_path)

    actions = []
    epoch_timesteps = []
    epoch_rewards = []
    dispatch = []
    timesteps = 0
    done = False
    while not done:
        action, processed_obs = agent.act(obs, eps=-1)
        next_obs, reward, done = env.step(action)
        print('timestep{} reward is {}'.format(timesteps, reward))
        next_obs_processed = agent.process_observation(next_obs)
        obs = next_obs
        timesteps += 1
        print('timestep{} action is {}'.format(timesteps, action))
        # print("reward：{}".format(reward))
        # print("timestep{} dispatch{}".format(timesteps, env.disp))
        epoch_rewards.append(reward)
        epoch_timesteps.append(timesteps)
        dispatch.append(env.disp)
        actions.append(action)
        if done:
            print('Test in timestep{} is done'.format(timesteps))
            break

    return epoch_rewards, epoch_timesteps, dispatch, agent, load, wind


def visualize_results(epoch_rewards, epoch_timesteps, dispatch, load, wind):
    # plt.plot(epoch_timesteps, epoch_rewards)
    # plt.show()
    dispatch_process = np.array(dispatch)
    print('wind power is {}'.format(wind))
    dispatch_all = np.sum(dispatch_process, axis=1) + wind
    epoch_rewards = np.sum(epoch_rewards)
    print('epoch reward is {}'.format(epoch_rewards))
    plt.bar(epoch_timesteps, dispatch_process[:, 0], alpha=0.5, label='G1')  # alpha:透明度
    plt.bar(epoch_timesteps, dispatch_process[:, 1], alpha=0.5, label='G2')
    plt.bar(epoch_timesteps, dispatch_process[:, 2], alpha=0.5, label='G3')
    plt.bar(epoch_timesteps, dispatch_process[:, 3], alpha=0.5, label='G4')
    plt.bar(epoch_timesteps, dispatch_process[:, 4], alpha=0.5, label='G5')
    plt.plot(epoch_timesteps, load, linestyle='dashed', label='load')
    plt.plot(epoch_timesteps, dispatch_all, alpha=0.5, label='dispatch')
    plt.xlabel('Period'), plt.ylabel('Output/Mxw')
    plt.legend(ncol=2, loc='best', framealpha=0.1)  # 为了让label显示
    plt.show()
    # print(dispatch)


epoch_rewards, epoch_timesteps, dispatch, agent, load, wind = test()
visualize_results(epoch_rewards, epoch_timesteps, dispatch, load, wind)
