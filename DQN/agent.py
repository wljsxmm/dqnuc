import random
from collections import deque
import torch
import torch.optim as optim
import numpy as np
from config import *
from networks import *
from rl4uc.environment import *



class ReplayMemory(object):

    def __init__(self, capacity, obs_size, act_dim):
        self.capacity = capacity
        self.obs_size = obs_size
        self.act_dim = act_dim

        self.act_buf = np.zeros((self.capacity, self.act_dim))
        self.obs_buf = np.zeros((self.capacity, self.obs_size))
        self.rew_buf = np.zeros(self.capacity)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_size))
        self.dones = np.zeros(self.capacity)
        self.num_used = 0
        self.store_num = 0
    def store(self, obs, action, reward, next_obs,dones):
        """Store a transition in the memory 前面把这些buf弄成全0矩阵 然后在main里面用while存入 """
        idx = self.num_used % self.capacity

        self.act_buf[idx] = action
        self.obs_buf[idx] = obs
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs
        self.dones[idx] = dones

        self.num_used += 1
        self.store_num +=1

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(self.capacity), size=batch_size, replace=False)
        # print('idx{}'.format(idx))
        data = {'act': self.act_buf[idx],
                'obs': self.obs_buf[idx],
                'rew': self.rew_buf[idx],
                'next_obs': self.next_obs_buf[idx],
                'dones':self.dones[idx]}
        # print('data{}'.format(data['act']))
        return data

    def is_full(self):
        return self.num_used >= self.capacity

    def reset_buffer(self):
        self.num_used = 0




class Agent:

    def __init__(self, env, bs, lr, tau, gamma, device):
        """
        When dealing with visual inputs, state_size should work as num_of_frame
        """
        # self.state_size = state_size
        self.num_gen = env.num_gen
        self.action_size = 2 * self.num_gen
        self.obs_size = self.process_observation(env.reset()).size
        self.bs = bs
        self.lr = lr
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.Q_local = Q_Network(self.obs_size, self.action_size).to(device)
        self.Q_target = Q_Network(self.obs_size, self.action_size).to(device)

        self.soft_update(1)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.Q_local.parameters(), self.lr)
        # self.memory = deque(maxlen=100000)
        self.MEMORY_SIZE = 300
        self.memory = ReplayMemory(self.MEMORY_SIZE, self.obs_size, self.num_gen)

    def process_observation(self,obs):
        obs_new = np.concatenate((obs['status'], [obs['timestep']], obs['wind_forecast'], obs['demand_forecast']))
        return obs_new

    def act(self, state, eps):  # epsilon
        processed_obs = self.process_observation(state)
        if random.random() > eps:
            processed_obs = torch.tensor(processed_obs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                action_values = self.Q_local(processed_obs)
                action_values = action_values.reshape(self.num_gen, 2)  # 这个返回最大动作值的trick要好好体会
            return action_values.argmax(axis=1).detach().numpy(), processed_obs
        else:
            # return random.choice(np.arange(self.action_size)), processed_obs
            return np.random.randint(0, 2, 5), processed_obs

    def update(self, memory, batch_size=None):
        if batch_size is None:
            batch_size = memory.capacity

        data = memory.sample(batch_size)

        qs = self.Q_local(data['obs']).reshape(batch_size, self.num_gen, 2)

        # A bit of complicated indexing here!
        # We are using the actions [batch_size, num_gen] to index Q-values
        # which have shape [batch_size, num_gen, 2]
        m, n = data['act'].shape
        I, J = np.ogrid[:m, :n]
        qs = qs[I, J, data['act']]

        next_qs = self.Q_target(data['next_obs']).reshape(batch_size, self.num_gen, 2)
        next_acts = next_qs.argmax(axis=2).detach().numpy()

        # The same complicated indexing!
        m, n = next_acts.shape
        I, J = np.ogrid[:m, :n]
        next_qs = next_qs[I, J, next_acts]

        # Recasting(重铸) rewards into the same shape as next_qs & The same as dones
        m, n = next_qs.shape
        rews = np.broadcast_to(data['rew'], (self.num_gen, batch_size)).T
        rews = torch.tensor(rews).float()  # 不用as_tensor shape : torch.Size([300, 5])
        dones = np.broadcast_to(data['dones'], (self.num_gen, batch_size)).T
        dones = torch.tensor(dones).float()
        with torch.no_grad():
            td_target = rews + self.gamma * next_qs*(1-dones)

        loss = self.criterion(qs, td_target)
        # 更新Q网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def learn(self):
        experiences = random.sample(self.memory, self.bs)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(self.device)

        Q_values = self.Q_local(states)
        Q_values = torch.gather(input=Q_values, dim=-1, index=actions)

        '这里是单步更新'
        with torch.no_grad():
            Q_targets = self.Q_target(next_states)
            Q_targets, _ = torch.max(input=Q_targets, dim=-1, keepdim=True)
            Q_targets = rewards + self.gamma * (1 - dones) * Q_targets  # 单步更新

        'Compute loss & Minimize the loss & 执行单个优化步骤'
        loss = (Q_values - Q_targets).pow(2).mean()  # 这里就是MSE的表达式
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, tau):
        for target_param, local_param in zip(self.Q_target.parameters(), self.Q_local.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def load_state_dict(self):
        self.Q_local = Q_Network.load_state_dict(torch.load('Data/{}_weights2.pth'.format(ENV_NAME)))
        pass
