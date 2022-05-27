import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib.pyplot as plt
from rl4uc.environment import *


class QAgent(nn.Module):
    def __init__(self, env):
        super(QAgent, self).__init__()
        self.num_gen = env.num_gen

        self.num_nodes = 32
        self.gamma = 0.99
        self.activation = torch.tanh

        # There are 2N output nodes, corresponding to ON/OFF for each generator
        self.n_out = 2 * self.num_gen

        self.obs_size = self.process_observation(env.reset()).size  # reset 返回的字典处理成 数组？

        self.in_layer = nn.Linear(self.obs_size, self.num_nodes)
        self.out_layer = nn.Linear(self.num_nodes, self.n_out)
        # 这里的weight 保存到哪里了 是否可以在test的时候调用训练好的神经网络

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()
        self.net = self.state_dict()
        # net1 = self.state_dict(torch.load('5_gens_weights.pth'))
        # torch.save(net, 'rl4uc/data/weights/{}_gens_weights.pth'.format(env.num_gen))

    def process_observation(self, obs):
        """
        Process an observation into a numpy array.

        Observations are given as dictionaries, which is not very convenient
        for function approximation. Here we take just the generator up/down times——status
        and the timestep.

        Customise this!
        """
        obs_new = np.concatenate((obs['status'], [obs['timestep']], obs['wind_forecast'], obs['demand_forecast']))
        # obs_new = np.concatenate((obs['status'], obs['timestep']))
        # [obs['demand_forecast', obs['wind_forecast']
        # print("next_obs_processed0{}".format(obs_new))
        return obs_new

    def forward(self, obs):
        x = torch.tensor(obs).float()
        x = self.activation(self.in_layer(x))
        return self.out_layer(x)

    def act(self, obs):
        """
        Agent always acts greedily w.r.t Q-values!
        """
        processed_obs = self.process_observation(obs)

        q_values = self.forward(processed_obs)
        q_values = q_values.reshape(self.num_gen, 2)  # 改成了一个num_gen*2形状的数组
        # print('q_values{}'.format(q_values))
        # if
        action = q_values.argmax(axis=1).detach().numpy()  # num_gen*2返回Q值大的 注意这里的0/1是下标索引 对应好开关机的操作
        ''' 
        1.detach() 返回一个新的tensor，从当前计算图中分离下来的，但是仍指向原变量的存放位置,
          不同之处只是requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad
        2.numpy() tensor to numpy
        '''
        # print('action{}'.format(action))
        return action, processed_obs

    def update(self, memory, batch_size=None):
        if batch_size is None:
            batch_size = memory.capacity

        data = memory.sample(batch_size)

        qs = self.forward(data['obs']).reshape(batch_size, self.num_gen, 2)

        # A bit of complicated indexing here!
        # We are using the actions [batch_size, num_gen] to index Q-values
        # which have shape [batch_size, num_gen, 2]
        m, n = data['act'].shape
        I, J = np.ogrid[:m, :n]
        qs = qs[I, J, data['act']]

        next_qs = self.forward(data['next_obs']).reshape(batch_size, self.num_gen, 2)
        next_acts = next_qs.argmax(axis=2).detach().numpy()

        # The same complicated indexing!
        m, n = next_acts.shape
        I, J = np.ogrid[:m, :n]
        next_qs = next_qs[I, J, next_acts]

        # Recasting rewards into the same shape as next_qs
        m, n = next_qs.shape
        rews = np.broadcast_to(data['rew'], (self.num_gen, batch_size)).T
        rews = torch.tensor(rews).float()  # 不用as_tensor

        td_target = rews + self.gamma * next_qs

        criterion = nn.MSELoss()
        loss = criterion(qs, td_target)
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print('nn.Linear.weight{}'.format(self.in_layer.weight))


class ReplayMemory(object):

    def __init__(self, capacity, obs_size, act_dim):
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_dim = act_dim

        self.act_buf = np.zeros((self.capacity, self.act_dim))
        self.obs_buf = np.zeros((self.capacity, self.obs_size))
        self.rew_buf = np.zeros(self.capacity)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_size))

        self.num_used = 0

    def store(self, obs, action, reward, next_obs):
        """Store a transition in the memory 前面把这些buf弄成全0矩阵 然后在main里面用while存入 """
        idx = self.num_used % self.capacity

        self.act_buf[idx] = action
        self.obs_buf[idx] = obs
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs

        self.num_used += 1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.capacity), size=batch_size, replace=False)
        # print('idx{}'.format(idx))
        data = {'act': self.act_buf[idx],
                'obs': self.obs_buf[idx],
                'rew': self.rew_buf[idx],
                'next_obs': self.next_obs_buf[idx]}
        print('data{}'.format(data['act']))
        return data

    def is_full(self):
        return (self.num_used >= self.capacity)

    def reset(self):
        self.num_used = 0


def train():
    render = 1  # 显示机组启停和出力  风力/负载  的一个柱状图 去展示这个环境
    MEMORY_SIZE = 200
    N_EPOCHS = 3500

    env = make_env_from_json('5gen')
    agent = QAgent(env)
    memory = ReplayMemory(MEMORY_SIZE, agent.obs_size, env.num_gen)

    log = {'mean_timesteps': [],
           'mean_reward': []}

    for i in range(N_EPOCHS):
        if i % 1 == 0:
            print("========================Epoch {}========================".format(i))
        epoch_timesteps = []
        epoch_rewards = []
        if render:
            pass
        while memory.is_full() == False:
            done = False
            obs = env.reset()
            timesteps = 0
            while not done:
                action, processed_obs = agent.act(obs)
                next_obs, reward, done = env.step(action)

                next_obs_processed = agent.process_observation(next_obs)

                memory.store(processed_obs, action, reward, next_obs_processed)

                obs = next_obs

                if memory.is_full():
                    print("memory is FULL!")
                    break

                timesteps += 1
                """测试 """
                print("timesteps：{}".format(timesteps))
                print("reward：{}".format(reward))
                # print("next_obs_processed1{}".format(next_obs_processed))

                if done:
                    print('epoch{}  '.format(i) + '  is done')
                    epoch_rewards.append(reward)
                    epoch_timesteps.append(timesteps)

        log['mean_timesteps'].append(np.mean(epoch_timesteps))
        log['mean_reward'].append(np.mean(epoch_rewards))

        agent.update(memory, batch_size=1)
        memory.reset()
    torch.save(agent.net, 'rl4uc/data/weights/{}_gens_weights3.pth'.format(env.num_gen))
    return agent, log

'最好是设置一个便捷的train 和 test 的开关'

agent, log = train()
pd.Series(log['mean_reward']).rolling(50).mean().plot()
plt.savefig('rl4uc/data/results/weight3.png')
plt.show()




def test():
    env = make_env(mode='test', profiles_df='test_data_5gen.csv')
    print(env.gen_info)
    print(env.episode_length)
    agent = QAgent(env)
    agent.load_state_dict(torch.load('5_gens_weights.pth'))

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
        print('timestep{} action is {}'.format(timesteps,action))
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

    return epoch_rewards, epoch_timesteps, dispatch, agent

'''
epoch_rewards, epoch_timesteps, dispatch, agent = test()
plt.plot(epoch_timesteps,epoch_rewards)
plt.show()
'''