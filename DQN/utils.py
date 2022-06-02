import numpy as np
from config import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    File_path = './Data/DDQN'
    # r = np.load(File_path+'/5gens_rewards2.npy')
    # r = np.load(File_path+'/5gens_average6-DDQN-L1-loss.npy')
    r = np.load(File_path+'/5gens_average_test1.npy')
    # r = np.load(File_path+'/5gens_average2.npy')
    # reward = []
    episode = NUM_EPISODE
    #
    # for i in range(episode):
    #     reward.append(r[i])


    plt.plot(np.arange(episode),r)
    plt.show()