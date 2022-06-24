import torch
import numpy as np
# device, CPU or the GPU of your choice
# GPU = 0
# DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cpu')
ACTION = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 1, 1],
                   [0, 0, 1, 0, 0],
                   [0, 0, 1, 0, 1],
                   [0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 1],
                   [0, 1, 0, 0, 0],
                   [0, 1, 0, 0, 1],
                   [0, 1, 0, 1, 0],
                   [0, 1, 0, 1, 1],
                   [0, 1, 1, 0, 0],
                   [0, 1, 1, 0, 1],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 1],
                   [1, 0, 0, 0, 0],
                   [1, 0, 0, 0, 1],
                   [1, 0, 0, 1, 0],
                   [1, 0, 0, 1, 1],
                   [1, 0, 1, 0, 0],
                   [1, 0, 1, 0, 1],
                   [1, 0, 1, 1, 0],
                   [1, 0, 1, 1, 1],
                   [1, 1, 0, 0, 0],
                   [1, 1, 0, 0, 1],
                   [1, 1, 0, 1, 0],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 0, 0],
                   [1, 1, 1, 0, 1],
                   [1, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1]])

gamma = 0.99
learning_rate = 1e-4
batch_size = 64
soft_update_freq = 60
capacity = 8
exploration = 256
epsilon_init = 1
epsilon_min = 0.01
decay = 0.999
n_iteration = 10

episode = 50
episode_wind = 50

n_step = 8
k_load = 6
interval_number = 21
