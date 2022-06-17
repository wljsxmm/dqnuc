# import numpy as np
# from rl4uc.environment import make_env_from_json
# from gym import envs
# env_specs = envs.registry.all()
# envs_ids = [env_spec.id for env_spec in env_specs]
# # print(envs_ids)
# import gym
# import rl4uc.environment
# import tianshou
# print(tianshou.__version__)
'''
env = gym.make('MountainCar-v0')
print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low,
        env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))
'''

import numpy as np
# print(np.__version__ )
# print(np.__path__)
# diff = 4
# dispatch_tolerance = 5
# ens_amount = diff if diff > dispatch_tolerance else 0
# print(ens_amount)
# print(np.random.randint(0, 2, 5))
# action = [1,0,1]
# idx = np.where(np.array(action) == 1)[0]
# print(idx)
#
# import itertools
#
# num = 0
# a = (0, 1)  # iterable  是元组
# for i in itertools.permutations(a, 2):
#     print(i)
#     num += 1
# print(num)


import platform
print(platform.platform())
import torch
print(torch.backends.mps.is_built())
print(torch.has_mps)