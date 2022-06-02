import torch
import torch.nn as nn
import torch.nn.functional as F


class DDQN_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden=[64, 256]):
        super(DDQN_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[0], hidden[1])

        self.value = nn.Linear(hidden[1], 1)
        self.adv = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        state = torch.as_tensor(state, dtype=torch.float)
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc2(y))
        adv = self.relu(self.fc3(y))

        value = self.value(value)
        adv = self.adv(adv)

        advAverage = torch.mean(adv, dim=0, keepdim=True)
        Q = value + adv - advAverage

        return Q


class Q_Network(nn.Module):

    def __init__(self, state_size, action_size, hidden=[64, 64]):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)

    def forward(self, state):
        x = torch.as_tensor(state, dtype=torch.float)
        # x = torch.tensor(state).float()
        # x = state.clone.detach().float()
        # x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
