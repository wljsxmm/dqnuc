import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import time
# import gym
from collections import deque
from rl4uc.environment import make_env,make_env_from_json
import matplotlib.pyplot as plt

'看看如何更改代码用以适应不同的算法  以便方便得去应用深度强化学习的框架去实验'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cpu')


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
    def __init__(self,observation_dim, gen_nums):
        super(DuelDdqn,self).__init__()
        self.action_dims = 2**gen_nums
        self.observation_dims = observation_dim
        self.fc = nn.Linear(self.observation_dims, 128)

        self.adv_fc1 = nn.Linear(128, 128)
        self.adv_fc2 = nn.Linear(128, self.action_dims)

        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, 1)

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


def train(buffer, target_model, eval_model, gamma, optimizer, batch_size, loss_fn, count, soft_update_freq, n_step):
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
    t1 = argmax_actions.unsqueeze(1)
    next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
    t2 = action.unsqueeze(1)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    expected_q_value = reward + (gamma ** n_step) * (1 - done) * next_q_value

    loss = loss_fn(q_value, expected_q_value.detach())
    # loss = (expected_q_value.detach() - q_value).pow(2)
    # loss = loss.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if count % soft_update_freq == 0:
        target_model.load_state_dict(eval_model.state_dict())


def action_process(action_index):
    ACTION = np.array([[0, 0, 0, 0, 0],
                       [1, 0, 0, 0, 0],
                       [0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0],
                       [0, 0, 1, 0, 0],
                       [1, 0, 1, 0, 0],
                       [0, 1, 1, 0, 0],
                       [1, 1, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [1, 0, 0, 1, 1],
                       [0, 1, 0, 1, 0],
                       [1, 1, 0, 1, 0],
                       [0, 0, 1, 1, 0],
                       [1, 0, 1, 1, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 0],
                       [0, 0, 0, 0, 1],
                       [0, 1, 0, 0, 1],
                       [1, 0, 0, 0, 1],
                       [1, 1, 0, 0, 1],
                       [0, 0, 1, 0, 1],
                       [0, 1, 1, 0, 1],
                       [1, 0, 1, 0, 1],
                       [1, 1, 1, 0, 1],
                       [0, 0, 0, 1, 1],
                       [0, 0, 1, 1, 1],
                       [0, 1, 0, 1, 1],
                       [0, 1, 1, 1, 1],
                       [1, 0, 0, 1, 1],
                       [1, 0, 1, 1, 1],
                       [1, 1, 0, 1, 1],
                       [1, 1, 1, 1, 1]]
                      )
    action = ACTION[action_index]
    return action


if __name__ == '__main__':
# def mainloop():
    gamma = 0.99
    learning_rate = 1e-3
    batch_size = 64
    soft_update_freq = 128
    capacity = 10000
    exploration = 256
    epsilon_init = 1
    epsilon_min = 0.01
    decay = 0.99
    episode = 30000
    n_step = 16
    render = False

    # env = gym.make('CartPole-v0')
    # env = env.unwrapped
    env = make_env_from_json('5gen')
    obs = env.reset()
    obs = np.concatenate(
        (obs['wind_forecast'], obs['demand_forecast']))
    observation_dim = obs.size
    # action_dim = 2*env.num_gen
    target_net = DuelDdqn(observation_dim, gen_nums=env.num_gen).to(device)
    eval_net = DuelDdqn(observation_dim, gen_nums=env.num_gen).to(device)
    eval_net.load_state_dict(target_net.state_dict())
    optimizer = torch.optim.Adam(eval_net.parameters(), lr=learning_rate)
    buffer = n_step_replay_buffer(capacity, n_step, gamma)
    loss_fn = nn.MSELoss()
    epsilon = epsilon_init
    count = 0
    episode_reward = []
    average_reward = []
    weighted_reward = []
    weight_reward = None

    time_begin = time.time()
    for i in range(episode):
        obs = env.reset()
        obs = np.concatenate(
            (obs['wind_forecast'], obs['demand_forecast']))
        if epsilon > epsilon_min:
            epsilon = epsilon * decay
        reward_total = 0

        while True:
            action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(device), epsilon=epsilon)
            count += 1
            # action = action_process(action_index[0])
            action = action_process(action_index)
            next_obs, reward, done = env.step(action)
            next_obs = np.concatenate(
                (next_obs['wind_forecast'], next_obs['demand_forecast']))
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
                print('episode: {}  epsilon: {:.2f}  reward: {}  weight_reward: {:.3f}'.format(i + 1, epsilon,
                                                                                               reward_total,
                                                                                               weight_reward))
                break
    time_end = time.time()

    eval_net.to('cpu')
    torch.save(eval_net.state_dict(), 'Data/NStepDDQN/weights_test9.pth')
    np.save('Data/NStepDDQN/reward_test9.npy', episode_reward)
    np.save('Data/NStepDDQN/average_test9.npy', average_reward)
    np.save('Data/NStepDDQN/weighted_test9.npy', weighted_reward)
    print('=============Total time is: {}============='.format(time_end-time_begin))
    plt.plot(np.arange(episode), average_reward)
    plt.plot(np.arange(episode), weighted_reward)
    plt.show()


def test():
    env = make_env(mode='test', profiles_df='test_data_10gen.csv')

    obs = env.reset()
    obs = np.concatenate(
        (obs['wind_forecast'], obs['demand_forecast']))
    observation_dim = obs.size
    load = env.episode_forecast
    wind = env.episode_wind_forecast

    parameter_path = 'Data/NStepDDQN/weights_test8.pth'
    # parameter_path = 'Data/{}_weights2.pth'.format(ENV_NAME)
    print(env.gen_info)
    print('Test episode length is {}'.format(env.episode_length_test))
    eval_net = DuelDdqn(observation_dim, gen_nums=env.num_gen)

    eval_net.load_state_dict(torch.load(parameter_path))

    actions = []
    epoch_timesteps = []
    epoch_rewards = []
    dispatch = []
    timesteps = 0
    done = False
    while not done:
        action_index = eval_net.act(torch.FloatTensor(np.expand_dims(obs, 0)), epsilon=-1)
        action = action_process(action_index)
        next_obs, reward, done = env.step(action)
        print('timestep{} reward is {}'.format(timesteps, reward))
        next_obs_processed = np.concatenate(
            (next_obs['wind_forecast'], next_obs['demand_forecast']))
        obs = next_obs_processed
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

    return epoch_rewards, epoch_timesteps, dispatch, load, wind


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


# epoch_rewards, epoch_timesteps, dispatch, load, wind = test()
# visualize_results(epoch_rewards, epoch_timesteps, dispatch, load, wind)
