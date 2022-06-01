import numpy as np
import matplotlib.pyplot as plt
File_path = './Data'
# r = np.load(File_path+'/5gen0_rewards.npy')
r = np.load(File_path+'/5gens_rewards1.npy')
# reward = []
episode = 20000
#
# for i in range(episode):
#     reward.append(r[i])


plt.plot(np.arange(episode),r)
plt.show()