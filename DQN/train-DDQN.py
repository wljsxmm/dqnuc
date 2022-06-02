import time

import numpy as np
# from utils import *  # 开始前的import为了能够让配置和一些预处理的函数  在主程序开始前就生效
from agent import *
from config import *
from rl4uc.environment import *
from tensorboardX import SummaryWriter
def 函数():  # python 竟然还能用中文定义函数名和变量名
    print('1111111111')
    pass

def train(agent, num_episode, eps_init, eps_decay, eps_min, max_t):
    writer = SummaryWriter('Logs/DDQN')
    rewards_log = []
    average_log = []
    eps = eps_init
    update_counter = 0

    for i in range(1, 1 + num_episode):
        episodic_reward = 0
        done = False
        state = env.reset()
        t = 0  # 这里的t 可以看成one episode 里面的 timestep

        print('Train episode length is {}'.format(env.episode_length))
        while not done and t < max_t:

            action, processed_obs = agent.act(state, eps)
            next_state, reward, done = env.step(action)
            next_state_processs = agent.process_observation(next_state)
            agent.memory.store(processed_obs, action, reward, next_state_processs, done)
            # state = next_state.copy()
            state = next_state

            if agent.memory.is_full():
                agent.memory.reset_buffer()

            t += 1
            update_counter += 1
            print("timesteps：{} take action:{} is finished ———————— reward: {}".format(t, action, reward))

            if update_counter % 4 == 0 and agent.memory.num_used >= agent.bs:
                loss = agent.update(agent.memory, BATCH_SIZE)
                print('Update steps {} :update the local net'.format(update_counter))
                writer.add_scalar('loss', loss.item(), global_step=update_counter)
            if update_counter % 10 == 0:
                agent.soft_update(agent.tau)
                '修改 target net update frequency'
                # if update_counter % 30 == 0:
                #     print('Update steps {} :update the target net'.format(update_counter))
                #     agent.soft_update(agent.tau)
                #     # agent.direct_update()

            episodic_reward += reward  # 记录一个episode的奖励

        rewards_log.append(episodic_reward)
        writer.add_scalar('episode reward', episodic_reward, global_step=i)
        average_log.append(np.mean(rewards_log[-100:]))
        print('\rEpisode {}, Reward {:.3f}, Average Reward {:.3f}'.format(i, episodic_reward, average_log[-1]))
        eps = max(eps * eps_decay, eps_min)
    return rewards_log, average_log, eps


'根据那个写test 和train 模式的切换'

if __name__ == '__main__':
    env = make_env_from_json('5gen')
    # env = make_env_from_json('5gen', profiles_df='data/test_data_5gen.csv')

    num_gens = env.num_gen
    states_reset = env.state
    agent = DDQN_Agent(num_gens, states_reset, BATCH_SIZE, LEARNING_RATE, TAU, GAMMA, DEVICE)
    time_start = time.time()
    rewards_log, average_log, eps = train(agent, NUM_EPISODE, EPS_INIT, EPS_DECAY, EPS_MIN, MAX_T)
    time_end = time.time()
    print('The final eps is: {} and total {} episodes time consuming is:{}'.format(eps, NUM_EPISODE , (time_end - time_start)))
    np.save('Data/DDQN/{}_reward_test1.npy'.format(ENV_NAME), rewards_log)
    np.save('Data/DDQN/{}_average_test1.npy'.format(ENV_NAME), average_log)

    agent.Q_local.to('cpu')
    torch.save(agent.Q_local.state_dict(), 'Data/DDQN/{}_weights_test1.pth'.format(ENV_NAME))

    '记录相关训练参数'
    result_dir = os.path.dirname(os.path.realpath(__file__))
    train_log_filename = "train_log.txt"
    train_log_filepath = os.path.join(result_dir, train_log_filename)
