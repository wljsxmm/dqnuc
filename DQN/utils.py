import numpy as np
from config import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    File_path = './Data/NStepDDQN'
    # r = np.load(File_path+'/5gens_rewards2.npy')
    # r = np.load(File_path+'/5gens_average6-DDQN-L1-loss.npy')
    average = np.load(File_path + '/average_test9.npy')
    weight = np.load(File_path+'/weighted_test9.npy')
    reward = np.load(File_path+'/reward_test9.npy')

    # r = np.load(File_path+'/5gens_average2.npy')
    # reward = []
    episode = 30000
    #
    # for i in range(episode):
    #     reward.append(r[i])

    plt.plot(np.arange(episode), average)
    # plt.plot(np.arange(2000,episode), average[2000:])

    # plt.ylim()
    # plt.plot(np.arange(episode), reward)
    # plt.plot(np.arange(episode), weight)  # 这里weight的处理 和average 的功能类似

    plt.show()
