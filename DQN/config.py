import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = torch.device("cuda:{}".format(GPU) if torch.cuda.is_available() else "cpu")

#environment names
ENV_NAME = '5gens'

#Agent parameters
BATCH_SIZE = 128
LEARNING_RATE = 0.0001
TAU = 0.001
GAMMA = 0.99

#Training parameters
NUM_EPISODE = 20000
EPS_INIT = 1
EPS_DECAY = 0.995
EPS_MIN = 0.05
MAX_T = 1500