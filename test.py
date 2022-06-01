import numpy as np
from rl4uc.environment import make_env_from_json
from gym import envs
env_specs = envs.registry.all()
envs_ids = [env_spec.id for env_spec in env_specs]
# print(envs_ids)
import gym
import rl4uc.environment
import tianshou
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


print(np.random.randint(0, 2, 5))
