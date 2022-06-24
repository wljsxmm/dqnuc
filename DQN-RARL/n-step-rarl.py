import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
import os
from config import *
from collections import deque
from rl4uc.environment import make_env, make_env_from_json
from environment_lite import env_make

import matplotlib.pyplot as plt

'看看如何更改代码用以适应不同的算法  以便方便得去应用深度强化学习的框架去实验'


class replay_buffer_wind(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*batch)
        return np.concatenate(observations, 0), actions, rewards, np.concatenate(next_observations, 0), dones

    def __len__(self):
        return len(self.memory)


class n_step_replay_buffer(object):

    def __init__(self, capacity, n_step, gamma):
        self.capacity = capacity
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.capacity)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward * (1 - do) + rew
            next_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def store(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.n_step_buffer.append([observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][: 2]
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        random.seed(1)
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(*batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)


class DuelDdqn(nn.Module):
    def __init__(self, observation_dim, gen_nums):
        super(DuelDdqn, self).__init__()
        self.action_dims = 2 ** gen_nums
        self.observation_dims = observation_dim
        self.fc = nn.Linear(self.observation_dims, 150)

        self.adv_fc1 = nn.Linear(150, 150)
        self.adv_fc2 = nn.Linear(150, self.action_dims)

        self.value_fc1 = nn.Linear(150, 150)
        self.value_fc2 = nn.Linear(150, 1)

        # self.fc1 = nn.Linear(observation_dim, 128)
        # self.activate = nn.Tanh()
        # self.FcValue = nn.Linear(128, 128)
        # self.FcAdv = nn.Linear(128, 128)
        # self.Value = nn.Linear(128, 1)
        # self.Adv = nn.Linear(128, self.action_dims)

    def forward(self, observation):
        feature = self.fc(observation)
        advantage = self.adv_fc2(F.relu(self.adv_fc1(F.relu(feature))))
        value = self.value_fc2(F.relu(self.value_fc1(F.relu(feature))))
        return advantage + value - advantage.mean()
        # y = self.activate(self.fc1(observation))
        # value = self.activate(self.FcValue(y))
        # adv = self.activate(self.FcAdv(y))
        #
        # value = self.Value(value)
        # adv = self.Adv(adv)
        #
        # adv_average = torch.mean(adv, dim=1, keepdim=True)
        # Q = value + adv - adv_average
        # return Q

    def act(self, observation, epsilon):
        random.seed(1)
        if random.random() > epsilon:
            q_value = self.forward(observation)
            # action_index = q_value.argmax(axis=1).detach().numpy()
            action_index = q_value.argmax(axis=1).detach().item()
            # action = action_process(action_index)
        else:
            action_index = np.random.randint(0, 32)
            # action = action_process(action_index)
        return action_index


class ddqn_wind(nn.Module):
    def __init__(self, observation_dim, interval_number):
        super(ddqn_wind, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = interval_number

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        # x = F.tanh(self.fc1(observation))
        x = F.relu(self.fc2(x))
        # x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, observation, epsilon):
        random.seed(1)
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action_index = q_value.argmax(axis=1).detach().item()
        else:
            action_index = np.random.randint(0, 20)
        return action_index


class ddqn(nn.Module):
    def __init__(self, observation_dim, gen_nums):
        super(ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.gen_nums = gen_nums
        self.action_dim = 2 ** gen_nums

        self.fc1 = nn.Linear(self.observation_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.action_dim)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        # x = F.tanh(self.fc1(observation))
        x = F.relu(self.fc2(x))
        # x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, observation, epsilon):
        random.seed(1)
        if random.random() > epsilon:
            q_value = self.forward(observation)
            action_index = q_value.argmax(axis=1).detach().item()
            # action = action_process(action_index)
        else:
            action_index = np.random.randint(0, 32)
            # action = action_process(action_index)
        return action_index


def train_wind(buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)

    observation = torch.FloatTensor(observation)
    action = torch.LongTensor(action)
    reward = torch.FloatTensor(reward)
    next_observation = torch.FloatTensor(next_observation)
    done = torch.FloatTensor(done)

    q_values = eval_model.forward(observation)
    next_q_values = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation).max(1)[1].detach()
    next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + gamma * (1 - done) * next_q_value

    loss = loss_fn(q_value, expected_q_value.detach())
    # loss = (expected_q_value.detach() - q_value).pow(2)
    # loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


def train(m, buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq, n_step):
    observation, action, reward, next_observation, done = buffer.sample(batch_size)
    observation = torch.FloatTensor(observation).to(device)
    action = np.array(action)
    action = torch.LongTensor(action).to(device)
    reward = torch.FloatTensor(reward).to(device)
    next_observation = torch.FloatTensor(next_observation).to(device)
    done = torch.FloatTensor(done).to(device)

    q_values = eval_model.forward(observation)
    next_q_values = target_model.forward(next_observation)
    argmax_actions = eval_model.forward(next_observation).max(1)[1].detach()
    # t1 = argmax_actions.unsqueeze(1)
    next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
    # t2 = action.unsqueeze(1)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + (gamma ** n_step) * (1 - done) * next_q_value

    loss = loss_fn(q_value, expected_q_value.detach())
    # loss = (expected_q_value.detach() - q_value).pow(2)
    # loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if m % soft_update_freq == 0:
        print('In episode {} soft update the target'.format(m))
        target_model.load_state_dict(eval_model.state_dict())


def action_process(action_index):
    action = ACTION[action_index]
    return action


def action_index_process(action):
    '''这个是修改过的，之前的权重都不能用了 对应的索引不再是对应的动作'''
    action_index = None
    for i in range(32):
        if (ACTION[i] == action).all():
            action_index = i
            break
    return action_index


def nsteppaper():
    parameter_path = os.path.dirname(os.path.realpath(__file__))
    env = env_make('train', 'train2016.csv')
    print(env.gen_info)
    buffer = n_step_replay_buffer(capacity, n_step, gamma)

    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    epsilon_wind = epsilon_init

    day = 1
    count = 0
    m_episodes = 1
    m = 1
    episode_reward = []
    observation_dim = (env.reset(day)).size
    target_net = DuelDdqn(observation_dim, gen_nums=env.num_gen).to(device)
    eval_net = DuelDdqn(observation_dim, gen_nums=env.num_gen).to(device)
    # target_net.load_state_dict(torch.load(parameter_path + '/Results//test2/paper/weights_generation.pth'))
    eval_net.load_state_dict(target_net.state_dict())

    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)

    # for i in range(m_episodes):
    while m <= m_episodes:
        obs = env.reset(day)

        reward_total = 0

        while True:
            count += 1
            action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device), epsilon=epsilon)
            action = action_process(action_index)
            real_action_index = action_index
            next_obs, reward, reward_wind, real_done, done = env.step(action, action_wind=(interval_number - 1) / 2)
            if env.change:
                real_action = env.commitment
                real_action_index = action_index_process(real_action)
            # TODO Store done or real_done?
            buffer.store(obs, real_action_index, reward, next_obs, done)
            reward_total += reward

            obs = next_obs

            if real_done:
                if buffer.__len__() >= batch_size :
                    train(m, buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq, n_step)
                    print('In count {} train the network'.format(count))
                episode_reward.append(reward_total)
                print('In episode {} day {} is over'.format(m,day))
                day += 1
                m += 1
                break
            elif done:

                if buffer.__len__() >= batch_size and count % (24 * batch_size) == 0:
                    train(m, buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq, n_step)
                    print('In count {}, the epsilon is {} and train the network'.format(count,epsilon))
                    if epsilon > epsilon_min:
                        epsilon = epsilon * decay
                break

            else:
                print('继续')

    eval_net.to('cpu')
    torch.save(eval_net.state_dict(), 'Results/test2/paper/weights_generation.pth')
    np.save('Results/test2/paper/episode_reward.npy', episode_reward)
    plt.plot(range(m_episodes), episode_reward)
    plt.show()


def mainloop(mode='rarl'):
    env = make_env_from_json('5gen')
    obs = env.reset()
    obs = np.concatenate((obs['wind_forecast'][0:k_load], obs['demand_forecast'][0:k_load]))
    observation_dim = obs.size

    target_net = ddqn(observation_dim, gen_nums=env.num_gen).to(device)
    eval_net = ddqn(observation_dim, gen_nums=env.num_gen).to(device)
    eval_net.load_state_dict(target_net.state_dict())

    target_net_wind = ddqn_wind(observation_dim, interval_number=interval_number).to(device)
    eval_net_wind = ddqn_wind(observation_dim, interval_number=interval_number).to(device)
    eval_net_wind.load_state_dict(target_net_wind.state_dict())

    buffer = n_step_replay_buffer(capacity, n_step, gamma)
    buffer_wind = replay_buffer_wind(capacity)

    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    epsilon_wind = epsilon_init

    count = 0
    episode_reward = []
    average_reward = []
    weighted_reward = []
    weight_reward = None

    episode_reward_wind = []
    average_reward_wind = []
    weighted_reward_wind = []
    weight_reward_wind = None
    time_begin = time.time()
    if mode == 'rarl':
        for i in range(n_iteration):

            for j in range(episode):

                obs = env.reset()
                obs = np.concatenate((obs['wind_forecast'][0:k_load], obs['demand_forecast'][0:k_load]))

                if epsilon > epsilon_min:
                    epsilon = epsilon * decay
                reward_total = 0
                index = 0

                while True:
                    action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device), epsilon=epsilon)
                    count += 1
                    index += 1
                    action = action_process(action_index)
                    action_wind_index = eval_net_wind.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device),
                                                          epsilon=-1)
                    next_obs, reward, reward_wind, done = env.step(action, action_wind=action_wind_index)

                    if index > 24 - k_load - 1:
                        next_obs = np.concatenate(
                            (next_obs['wind_forecast'][index:24], next_obs['wind_forecast'][0:(index + k_load) % 24],
                             next_obs['demand_forecast'][index:24],
                             next_obs['demand_forecast'][0:(index + k_load) % 24]))
                    else:
                        next_obs = np.concatenate((next_obs['wind_forecast'][index:index + k_load],
                                                   next_obs['demand_forecast'][index:index + k_load]))

                    buffer.store(obs, action_index, reward, next_obs, done)
                    reward_total += reward
                    obs = next_obs

                    if j > exploration:
                        train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count,
                              soft_update_freq, n_step)

                    if done:
                        episode_reward.append(reward_total)
                        average_reward.append(np.mean(episode_reward[-100:]))
                        if not weight_reward:
                            weight_reward = reward_total
                            weighted_reward.append(weight_reward)
                        else:
                            weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                            weighted_reward.append(weight_reward)
                        print(
                            'In Iteration:{} Protagonist episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {'
                            ':.3f}'.format(
                                i + 1, j + 1, epsilon,
                                reward_total,
                                weight_reward))
                        break

            eval_net.to('cpu')
            torch.save(eval_net.state_dict(), 'Results/test2/rarl/weights_generations{}.pth'.format(i + 1))
            np.save('Results/test2/rarl/generations{}.npy'.format(i + 1), episode_reward)
            np.save('Results/test2/rarl/average_generations{}.npy'.format(i + 1), average_reward)
            np.save('Results/test2/rarl/weighted_generations{}.npy'.format(i + 1), weighted_reward)

            for k in range(episode_wind):
                obs = env.reset()
                obs = np.concatenate((obs['wind_forecast'][0:k_load], obs['demand_forecast'][0:k_load]))

                if epsilon_wind > epsilon_min:
                    epsilon_wind = epsilon_wind * decay

                reward_total_wind = 0
                index = 0
                while True:
                    action_index = eval_net_wind.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device),
                                                     epsilon=epsilon_wind)
                    count += 1
                    index += 1

                    action_generation_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device),
                                                           epsilon=-1)
                    action_generation = action_process(action_generation_index)
                    next_obs, reward, reward_wind, done = env.step(action_generation, action_wind=action_index)

                    if index > 24 - k_load - 1:
                        next_obs = np.concatenate(
                            (next_obs['wind_forecast'][index:24], next_obs['wind_forecast'][0:(index + k_load) % 24],
                             next_obs['demand_forecast'][index:24],
                             next_obs['demand_forecast'][0:(index + k_load) % 24]))
                    else:
                        next_obs = np.concatenate((next_obs['wind_forecast'][index:index + k_load],
                                                   next_obs['demand_forecast'][index:index + k_load]))

                    buffer_wind.store(obs, action_index, reward_wind, next_obs, done)
                    reward_total_wind += reward_wind
                    obs = next_obs

                    if k > exploration:
                        train_wind(buffer_wind, target_net_wind, eval_net_wind, gamma, optimizer, batch_size, loss_fn,
                                   count, soft_update_freq)
                    if done:
                        episode_reward_wind.append(reward_total_wind)
                        average_reward_wind.append(np.mean(episode_reward_wind[-100:]))
                        if not weight_reward_wind:
                            weight_reward_wind = reward_total_wind
                            weighted_reward_wind.append(weight_reward_wind)
                        else:
                            weight_reward_wind = 0.99 * weight_reward_wind + 0.01 * reward_total_wind
                            weighted_reward_wind.append(weight_reward_wind)
                        print('In Iteration: {}  Adversary episode: {} reward: {}  weight_reward: {:.3f}'.format(i + 1,
                                                                                                                 k + 1,
                                                                                                                 reward_total_wind,
                                                                                                                 weight_reward_wind))
                        break

            eval_net_wind.to('cpu')
            torch.save(eval_net_wind.state_dict(), 'Results/test2/rarl/weights_wind{}.pth'.format(i + 1))
            np.save('Results/test2/rarl/reward_wind{}.npy'.format(i + 1), episode_reward_wind)
            np.save('Results/test2/rarl/average_wind{}.npy'.format(i + 1), average_reward_wind)
            np.save('Results/test2/rarl/weighted_wind{}.npy'.format(i + 1), weighted_reward_wind)
            plt.plot(np.arange(episode * (i + 1)), average_reward)
            plt.plot(np.arange(episode_wind * (i + 1)), average_reward_wind)
            plt.plot(np.arange(episode * (i + 1)), weighted_reward)
            plt.plot(np.arange(episode_wind * (i + 1)), weighted_reward_wind)
            plt.show()

        torch.save(eval_net.state_dict(), 'Results/test2/rarl/total/weights_wind{}.pth'.format(i + 1))
        torch.save(eval_net_wind.state_dict(), 'Results/test2/rarl/total/weights_wind{}.pth'.format(i + 1))

        np.save('Results/test2/rarl/total/reward_generations{}.npy'.format(i + 1), episode_reward)
        np.save('Results/test2/rarl/total/average_generations{}.npy'.format(i + 1), average_reward)
        np.save('Results/test2/rarl/total/weighted_generations{}.npy'.format(i + 1), weighted_reward)

        np.save('Results/test2/rarl/total/reward_wind{}.npy'.format(i + 1), episode_reward_wind)
        np.save('Results/test2/rarl/total/average_wind{}.npy'.format(i + 1), average_reward_wind)
        np.save('Results/test2/rarl/total/weighted_wind{}.npy'.format(i + 1), weighted_reward_wind)

        time_end = time.time()
        print('=============Total time is: {}============='.format(time_end - time_begin))

    else:
        for i in range(episode):

            obs = env.reset()
            obs = np.concatenate((obs['wind_forecast'][0:k_load], obs['demand_forecast'][0:k_load]))

            if epsilon > epsilon_min:
                epsilon = epsilon * decay
            reward_total = 0
            index = 0

            while True:
                action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device), epsilon=epsilon)
                count += 1
                index += 1
                action = action_process(action_index)
                # action_wind_index = eval_net_wind.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device),
                #                                       epsilon=-1)
                next_obs, reward, reward_wind, done = env.step(action, action_wind=(interval_number - 1) / 2)

                if index > 24 - k_load - 1:
                    next_obs = np.concatenate(
                        (next_obs['wind_forecast'][index:24], next_obs['wind_forecast'][0:(index + k_load) % 24],
                         next_obs['demand_forecast'][index:24],
                         next_obs['demand_forecast'][0:(index + k_load) % 24]))
                else:
                    next_obs = np.concatenate((next_obs['wind_forecast'][index:index + k_load],
                                               next_obs['demand_forecast'][index:index + k_load]))

                buffer.store(obs, action_index, reward, next_obs, done)
                reward_total += reward
                obs = next_obs

                if i > exploration:
                    train(buffer, target_net, eval_net, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq,
                          n_step)

                if done:
                    episode_reward.append(reward_total)
                    average_reward.append(np.mean(episode_reward[-100:]))
                    if not weight_reward:
                        weight_reward = reward_total
                        weighted_reward.append(weight_reward)
                    else:
                        weight_reward = 0.99 * weight_reward + 0.01 * reward_total
                        weighted_reward.append(weight_reward)
                    print(
                        'Training vanilla agent! Episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(
                            i + 1, epsilon,
                            reward_total,
                            weight_reward))
                    break
            if (i % 500 == 0) and i > 0:
                plt.plot(np.arange(i + 1), average_reward)
                plt.plot(np.arange(i + 1), weighted_reward)
                plt.show()

        eval_net.to('cpu')
        torch.save(eval_net.state_dict(), 'Results/test2/vanilla/weights_generation.pth')
        np.save('Results/test2/vanilla/reward_generation.npy', episode_reward)
        np.save('Results/test2/vanilla/average_generation.npy', average_reward)
        np.save('Results/test2/vanilla/weighted_generation.npy', weighted_reward)


def test():
    env = make_env(mode='test', profiles_df='test_data_10gen.csv')
    k_load = 6
    obs = env.reset()
    obs = np.concatenate((obs['wind_forecast'][0:k_load], obs['demand_forecast'][0:k_load]))
    observation_dim = obs.size
    load = env.episode_forecast
    wind = env.episode_wind_forecast
    # wind = 0
    parameter_path = '/Users/xmm/PycharmProjects/pythonProject/rl4uc-master/DQN-RARL/Results/test2/rarl/weights_generations49.pth'
    # parameter_path = '/Users/xmm/PycharmProjects/pythonProject/rl4uc-master/DQN-RARL/Results/test2/vanilla/weights_generation.pth'

    print('Test episode length is {}'.format(env.episode_length_test))
    eval_net = ddqn(observation_dim, gen_nums=env.num_gen)

    eval_net.load_state_dict(torch.load(parameter_path))

    actions = []
    dispatch = []

    epoch_timesteps = []
    epoch_rewards = []

    timesteps = 0
    done = False

    while not done:
        action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon=-1)
        action = action_process(action_index)
        next_obs, reward, _, done = env.step(action)
        print('timestep{} reward is {}'.format(timesteps, reward))

        if timesteps > 24 - k_load - 1:
            next_obs = np.concatenate(
                (next_obs['wind_forecast'][timesteps:24], next_obs['wind_forecast'][0:(timesteps + k_load) % 24],
                 next_obs['demand_forecast'][timesteps:24], next_obs['demand_forecast'][0:(timesteps + k_load) % 24]))
        else:
            next_obs = np.concatenate(
                (next_obs['wind_forecast'][timesteps:timesteps + k_load],
                 next_obs['demand_forecast'][timesteps:timesteps + k_load]))

        obs = next_obs
        timesteps += 1
        # print('timestep{} action is {}'.format(timesteps, action))
        # print("reward：{}".format(reward))
        # print("timestep{} dispatch{}".format(timesteps, env.disp))
        epoch_rewards.append(reward)
        epoch_timesteps.append(timesteps)
        dispatch.append(env.disp)
        actions.append(action)
        if done:
            print('Test in timestep{} is done'.format(timesteps))
            break

    return epoch_rewards, epoch_timesteps, dispatch, load, wind, actions


def visualize_results(epoch_rewards, epoch_timesteps, dispatch, load, wind, actions):
    # plt.plot(epoch_timesteps, epoch_rewards)
    # plt.show()
    print('The schedule of the test day is {}'.format(np.array(actions)))
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


if __name__ == '__main__':
    # mainloop('rarl')
    # mainloop('vanilla')
    nsteppaper()
    # epoch_rewards, epoch_timesteps, dispatch, load, wind, actions = test()
    # visualize_results(epoch_rewards, epoch_timesteps, dispatch, load, wind, actions)
