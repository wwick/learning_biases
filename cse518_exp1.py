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
    agent = FastMyopicAgent(horizon=myopic_horizon)
    optimal_agent = FastMyopicAgent(horizon=episode_length)
    data = np.zeros((num_grids,4,width,width))

    for i in range(num_grids):
        gridworld = gen_random_connected(width, width, num_rewards)
        mdp = Mdp(gridworld)

        start_state = gridworld.get_random_start_state()
        mdp.gridworld.start_state = start_state

        agent.set_mdp(gridworld)
        optimal_agent.set_mdp(gridworld)

        agent_action = agent.get_action(start_state)
        optimal_action = optimal_agent.get_action(start_state)

        r1,r2 = 0.0,0.0

        if agent_action != optimal_action:
            agent_trajectory = run_agent(agent,mdp,episode_length=episode_length)
            r1 = get_reward_from_trajectory(agent_trajectory)
            intervened_trajectory = run_agent(agent,mdp,episode_length=episode_length, first_optimal=optimal_agent)
            r2 = get_reward_from_trajectory(intervened_trajectory)
        data[i,:3] = gen_gridworld_arr(gridworld, width)
        data[i,3] = r2 - r1
    return data


def get_all_data(width, num_rewards, episode_length, myopic_horizon):
    pos,neg = 0,0
    data = np.zeros((0,4,width,width))
    while pos < 120000 or neg < 120000:
        with Pool(8) as p:
            n = 1000
            run_data = p.starmap(gen_data, [[100,width,num_rewards,episode_length,myopic_horizon]]*n)
            run_data = np.concatenate(run_data)
            data = np.concatenate((data, run_data))
            y = data[:,3,0,0]
            y = (y>10).astype(int)
            pos = (y==1).sum()
            neg = (y==0).sum()
            print(pos, neg)
    x = data[:,:3]
    y = data[:,3,0,0]
    data_path = f'/scratch1/fs1/chien-ju.ho/RIS/518/scripts/{width}_{num_rewards}_{episode_length}_{myopic_horizon}'
    np.savez_compressed(data_path, x=x, y=y)

    # return xtrain, ytrain, xeval, yeval

    
if __name__ == '__main__':
    size = int(sys.argv[1])
    reward = int(sys.argv[2])
    episode = int(sys.argv[3])
    myopic = int(sys.argv[4])
    get_all_data(size, reward, episode, myopic)