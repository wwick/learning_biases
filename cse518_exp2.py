from gridworld import GridworldMdp
from agents import OptimalAgent, MyopicAgent
from fast_agents import FastMyopicAgent, FastOptimalAgent
from mdp_interface import Mdp
from agent_runner import get_reward_from_trajectory, run_agent
import numpy as np
from maia_chess_backend.maia.tfprocess import get_tfp
import tensorflow as tf
from multiprocessing import Pool
import tqdm
import sys

def gen_gridworld_arr(gridworld, width):
    arr = np.zeros((3,width,width), dtype=np.int8)
    arr[0] = np.array(gridworld.walls)
    
    for (x,y) in gridworld.rewards:
        arr[1,x,y] = gridworld.rewards[(x,y)]
        
    (x,y) = gridworld.get_start_state()
    arr[2,x,y] = 1
    
    return arr

def gen_random_connected(width, height, num_rewards):
    for _ in range(1000):
        try:
            return GridworldMdp.generate_random_connected(width=width,height=height,num_rewards=num_rewards,noise=0)
        except:
            pass
    raise ValueError('Could not generate Gridworld')
    

def gen_data(num_grids, width, num_rewards, episode_length, myopic_horizon):
    all_data = np.zeros((num_grids,4,width,width))
    all_rewards = np.zeros((num_grids,2**episode_length,2))
    for i in range(num_grids):
        gridworld = gen_random_connected(width, width, num_rewards)
        mdp = Mdp(gridworld)
        start_state = gridworld.get_start_state()
        mdp.gridworld.start_state = start_state
        dummy_agent = FastMyopicAgent(horizon=episode_length)
        dummy_agent.set_mdp(gridworld)
        def recurse(agent_list, moves_left):
            if moves_left == 0: return [agent_list]
            myopic_agent = FastMyopicAgent(horizon=min(moves_left, myopic_horizon))
            optimal_agent = FastMyopicAgent(horizon=moves_left)
            l1 = recurse(agent_list+[(0,myopic_agent)], moves_left-1)
            l2 = recurse(agent_list+[(1,optimal_agent)], moves_left-1)
            return l1+l2
        agent_lists = recurse([], episode_length)
        rewards = []
        for agent_list in agent_lists:
            num_ints = sum([j[0] for j in agent_list])
            agent_list = [j[1] for j in agent_list]
            trajectory = run_agent(dummy_agent, mdp, episode_length=episode_length, agent_list=agent_list)
            rewards.append((get_reward_from_trajectory(trajectory),num_ints))
        rewards = np.array(rewards)
        all_data[i,:3] = gen_gridworld_arr(gridworld, width)
        all_rewards[i,] = rewards
    return all_data, all_rewards


def get_all_data(width, num_rewards, episode_length, myopic_horizon):
    iters = 200
    all_data = np.zeros((0,4,width,width))
    all_rewards = np.zeros((0,2**episode_length,2))
    for _ in range(iters):
        with Pool(32) as p:
            n = 1000
            run_data = p.starmap(gen_data, [[1,width,num_rewards,episode_length,myopic_horizon]]*n)

            datas = [d[0] for d in run_data]
            datas = np.concatenate(datas)
            all_data = np.concatenate((all_data, datas))


            rewards = [d[1] for d in run_data]
            rewards = np.concatenate(rewards)
            all_rewards = np.concatenate((all_rewards, rewards))
            
            print(all_data.shape, all_rewards.shape)


#                 y = data[:,3,0,0]
#                 pos = (y==1).sum()
#                 neg = (y==0).sum()
#                 print(pos, neg)
#         x = data[:,:3]
#         y = data[:,3,0,0]
        data_path = f'/scratch1/fs1/chien-ju.ho/RIS/518/exp2/{width}_{num_rewards}_{episode_length}_{myopic_horizon}'
        np.savez_compressed(data_path, all_data=all_data, all_rewards=all_rewards)

# return xtrain, ytrain, xeval, yeval

    
if __name__ == '__main__':
    size = int(sys.argv[1])
    reward = int(sys.argv[2])
    episode = int(sys.argv[3])
    myopic = int(sys.argv[4])
#     data, rewards = gen_data(10,size,reward,episode,myopic,0)
#     print('data', data.sum())
#     print(data)
#     print('rewards')
#     print(rewards)
    get_all_data(size, reward, episode, myopic)