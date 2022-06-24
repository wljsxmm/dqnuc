#!/usr/bin/env python3
import array
import random

import numpy as np
import pandas as pd
import os
import json
from scipy.stats import weibull_min

from rl4uc.dispatch import lambda_iteration

DEFAULT_PROFILES_FN = 'data/train_data_10gen.csv'

DEFAULT_VOLL = 40000
DEFAULT_EPISODE_LENGTH_HRS = 24
DEFAULT_DISPATCH_RESOLUTION = 1
DEFAULT_DISPATCH_FREQ_MINS = 60
DEFAULT_MIN_REWARD_SCALE = -5000
DEFAULT_NUM_GEN = 5
DEFAULT_EXCESS_CAPACITY_PENALTY_FACTOR = 0
DEFAULT_STARTUP_MULTIPLIER = 1


class Env(object):

    def __init__(self, gen_info, profiles_df, mode='train', **kwargs):

        self.availability = None
        self.real_done = None
        self.change = None
        self.forecast = None
        self.wind_forecast = None
        self.mode = mode  # Test or train. Determines the reward function and is_terminal()
        self.gen_info = gen_info
        self.profiles_df = profiles_df

        self.voll = kwargs.get('voll', DEFAULT_VOLL)  # value of lost load
        self.dispatch_freq_mins = DEFAULT_DISPATCH_FREQ_MINS  # Dispatch frequency in minutes
        self.dispatch_resolution = self.dispatch_freq_mins / 60.
        self.num_gen = self.gen_info.shape[0]

        self.episode_length = DEFAULT_EPISODE_LENGTH_HRS

        # Generator info 
        self.max_output = self.gen_info['max_output'].to_numpy()
        self.min_output = self.gen_info['min_output'].to_numpy()
        self.status = self.gen_info['status'].to_numpy()
        self.a = self.gen_info['a'].to_numpy()
        self.b = self.gen_info['b'].to_numpy()
        self.c = self.gen_info['c'].to_numpy()
        self.t_min_down = self.gen_info['t_min_down'].to_numpy()
        self.t_min_up = self.gen_info['t_min_up'].to_numpy()
        self.t_max_up = self.gen_info['t_max_up'].to_numpy()
        self.hot_cost = self.gen_info['hot_cost'].to_numpy()
        self.cold_cost = self.gen_info['cold_cost'].to_numpy()
        self.cold_hrs = self.gen_info['cold_hrs'].to_numpy()

        # Min and max demand for clipping demand profiles
        self.min_demand = np.max(self.min_output)
        self.max_demand = np.sum(self.max_output)

        # Tolerance parameter for lambda-iteration 
        self.dispatch_tolerance = 1  # epsilon for lambda iteration.

        self.start_cost = 0
        self.day_cost = 0  # cost for the entire day

        self.action_space = self.num_gen

    def _determine_constraints(self):
        """
        Determine which generators must be kept on or off for the next time period.
        """
        self.must_on = np.array([True if 0 < self.status[i] < self.t_min_up[i] else False for i in range(self.num_gen)])
        self.must_off = np.array(
            [True if -self.t_min_down[i] < self.status[i] < 0 else False for i in range(self.num_gen)])

    def _legalise_action(self, action):
        """
        Convert an action to be legal (remaining the same if it is already legal)
        Considers constraints set in self.determine_constraints()
        """
        x = np.logical_or(np.array(action), self.must_on)
        x = x * np.logical_not(self.must_off)
        return (np.array(x, dtype=int))

    def _is_legal(self, action):

        action = np.array(action)
        illegal_on = np.any(action[self.must_on] == 0)
        illegal_off = np.any(action[self.must_off] == 1)
        if any([illegal_on, illegal_off]):
            return False
        else:
            return True

    def _get_net_demand(self, deterministic, errors, curtail=False):

        demand_error = 0
        wind_error = 0

        demand_real = self.forecast + demand_error
        self.demand_real = max(0, demand_real)

        wind_real = self.wind_forecast + wind_error
        self.wind_real = max(0, wind_real)

        net_demand = demand_real - wind_real

        return net_demand

    def roll_forecasts(self, action_wind):

        # print('Run roll_forecasts')
        self.forecast = self.episode_forecast[self.episode_timestep]
        factor = (2 * action_wind - 20) / 100
        self.wind_forecast = (1 + factor) * self.episode_wind_forecast[self.episode_timestep]
        self.episode_timestep += 1

    def update_gen_status(self, action):
        def single_update(status, action):
            if status > 0:
                if action == 1:
                    return (status + 1)
                else:
                    return -1
            else:
                if action == 1:
                    return 1
                else:
                    return (status - 1)

        self.status = np.array([single_update(self.status[i], action[i]) for i in range(len(self.status))])

    def calculate_lost_load_cost(self, net_demand, disp, availability=None):
        diff = abs(net_demand - np.sum(disp))
        ens_amount = diff if diff > self.dispatch_tolerance else 0
        ens_cost = ens_amount * self.voll * self.dispatch_resolution
        return ens_cost

    def get_state_paper(self):

        state = {'timestep': self.episode_timestep,
                 'action': self.commitment,
                 'status': self.status,
                 'dispatch': self.disp,
                 'demand_forecast': self.episode_forecast,
                 'wind_forecast': self.episode_wind_forecast,
                 }
        # self.state = state
        state = np.concatenate(([state['timestep']], state['action'], state['action'], state['dispatch'],
                                state['demand_forecast'], state['wind_forecast']))
        return state

    def step(self, action, deterministic=True, errors=None, action_wind=0):

        obs = self.transition_paper(action, deterministic, errors, action_wind)
        reward = self._get_reward()  # Evaluate the reward function
        reward_wind = -reward
        real_done, done = self.is_terminal()  # Determine if state is terminal

        return obs, reward, reward_wind,real_done ,done

    def transition_paper(self, action, deterministic, errors, actions_wind):

        if self._is_legal(action) is False:
            commitment_action = self._legalise_action(action)
        else:
            commitment_action = action

        if (commitment_action == np.array(action)).all():
        # if commitment_action.all() == np.array(action).all():
            self.change = False
        else:
            print('commitment_action is {}, action is {}'.format(commitment_action,action))
            print('已改换动作')
            self.change = True
        # TODO

        # timestep + 1
        self.roll_forecasts(actions_wind)

        self.net_demand = self._get_net_demand(deterministic, errors)

        self.commitment = np.array(commitment_action)

        '截止到 t- 的运行时间'
        self.update_gen_status(self.commitment)

        # 更新self.must_on 的程序
        self._determine_constraints()

        self.start_cost = self._calculate_start_costs()
        self.fuel_costs, self.disp = self.calculate_fuel_cost_and_dispatch(self.net_demand, commitment_action,
                                                                           self.availability)

        self.fuel_cost = np.sum(self.fuel_costs)
        self.ens_cost = self.calculate_lost_load_cost(self.net_demand, self.disp, self.availability)
        self.ens = True if self.ens_cost > 0 else False

        self.day_cost += self.start_cost + self.fuel_cost + self.ens_cost

        state = self.get_state_paper()

        return state

    def _get_reward(self):
        """Calculate the reward (negative operating cost)"""
        operating_cost = self.fuel_cost + self.ens_cost + self.start_cost
        reward = -operating_cost
        self.reward = reward
        return reward

    def is_terminal(self):

        if self.mode == "train":
            if self.ens:
                real_done = False
                self.real_done = real_done
                done = True
                return real_done, done
            elif (self.ens is False) and self.episode_timestep == (self.episode_length - 1):
                real_done = True
                done = True
                self.real_done = real_done
                return real_done, done
            else:
                real_done = False
                self.real_done = real_done
                done = False
                return real_done, done
        else:
            print('run self.episode_timestep {}'.format(self.episode_timestep))
            return (self.episode_timestep == (self.episode_length - 1)) or self.ens

    def economic_dispatch(self, action, demand, lambda_lo, lambda_hi):
        idx = np.where(np.array(action) == 1)[0]
        on_a = self.a[idx]
        on_b = self.b[idx]
        on_min = self.min_output[idx]
        on_max = self.max_output[idx]
        disp = np.zeros(self.num_gen)
        if np.sum(on_max) < demand:
            econ = on_max
        elif np.sum(on_min) > demand:
            econ = on_min
        else:
            econ = lambda_iteration(demand, lambda_lo,
                                    lambda_hi, on_a, on_b,
                                    on_min, on_max, self.dispatch_tolerance)
        disp[idx] = econ

        return disp

    def _generator_fuel_costs(self, output, commitment):
        costs = np.multiply(self.a, np.square(output)) + np.multiply(self.b, output) + self.c
        costs = costs * self.dispatch_resolution  # Convert to MWh by multiplying by dispatch resolution in hrs
        costs = costs * commitment
        return costs

    def _calculate_fuel_costs(self, output, commitment):
        """ 
        Calculate total fuel costs for each generator, returning the sum.

        The fuel costs are quadratic: C = ax^2 + bx + c
        """
        costs = self._generator_fuel_costs(output, commitment)
        return costs

    def _calculate_start_costs(self):
        idx = np.where(self.status == 1)[0]
        start_cost = np.sum(self.hot_cost[idx])  # only hot costs
        return start_cost

    def calculate_fuel_cost_and_dispatch(self, demand, commitment, availability=None):

        disp = self.economic_dispatch(commitment, demand, 0, 100)
        fuel_costs = self._calculate_fuel_costs(disp, commitment)
        return fuel_costs, disp

    def choose_day(self, day_rounds):

        day = self.profiles_df.date[0 + 48 * (day_rounds - 1)]
        print('The sampled day is:{}'.format(day))
        day_profile = self.profiles_df[self.profiles_df.date == day]
        return day, day_profile

    def reset(self, day_number):

        day_initial_state = self.status

        if self.mode == 'train':
            day, day_profile = self.choose_day(day_number)
            self.day = day
            self.episode_forecast = day_profile.demand.values[0:48:2]
            self.episode_wind_forecast = day_profile.wind.values[0:48:2]
        else:
            print('还没想好')
            pass

        self.episode_timestep = 0
        self.forecast = None
        self.wind_forecast = None

        if (not self.real_done) and day_number > 1:
            self.status = day_initial_state

        self.net_demand = None
        self.day_cost = 0

        self.commitment = np.where(self.status > 0, 1, 0)
        self._determine_constraints()

        self.roll_forecasts(action_wind=10)

        self.net_demand = self._get_net_demand(deterministic=True, errors=None)
        _, self.disp = self.calculate_fuel_cost_and_dispatch(self.net_demand,self.commitment,availability=None)

        # self.update_gen_status(self.commitment)

        self.ens = False

        state = self.get_state_paper()

        return state


def create_gen_info(num_gen, dispatch_freq_mins):
    """
    Create a gen_info data frame for number of generators and dispatch frequency. 
    Created by copying generators kazarlis 10 gen problem. 
    
    The 10 generator problem is for 30 minute resolution data, so min_down_time
    status, and other time-related vars need to be scaled accordingly.
    
    """
    MIN_GENS = 5
    if num_gen < 5:
        raise ValueError("num_gen should be at least {}".format(MIN_GENS))

    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Repeat generators
    gen10 = pd.read_csv(os.path.join(script_dir,
                                     'data/kazarlis_units_10.csv'))
    gen5 = pd.read_csv(os.path.join(script_dir,
                                    'data/kazarlis_units_5.csv'))
    if num_gen == 5:
        gen_info = gen5
        # gen_info = gen10[::2]  # Special: for 5 gens, take every other generator
    else:
        upper_limit = int(np.floor(num_gen / 10) + 1)
        gen_info = pd.concat([gen10] * upper_limit)[:num_gen]

    gen_info = gen_info.sort_index()
    gen_info.reset_index()

    # Scale time-related variables  缩放时间相关变量
    gen_info.t_min_up = gen_info.t_min_up * (60 / dispatch_freq_mins)  # 因为在csv文件里的min的单位是h 所以要将
    gen_info.t_min_down = gen_info.t_min_down * (60 / dispatch_freq_mins)
    gen_info.status = gen_info.status * (60 / dispatch_freq_mins)
    gen_info = gen_info.astype({'t_min_down': 'int64',  # 把这些变量变成整型int变量
                                't_min_up': 'int64',
                                'status': 'int64'})

    return gen_info


def env_make(mode='train', profiles_df=None, **params):
    """
    用于连续天数训练的环境
    """
    script_dir = os.path.dirname(os.path.realpath(__file__))
    gen_info = create_gen_info(params.get('num_gen', DEFAULT_NUM_GEN),
                               params.get('dispatch_freq_mins', DEFAULT_DISPATCH_FREQ_MINS))

    profiles_df = pd.read_csv(os.path.join(script_dir + '/data', profiles_df))

    env = Env(gen_info=gen_info, profiles_df=profiles_df, mode=mode, **params)
    # env.reset()
    return env
