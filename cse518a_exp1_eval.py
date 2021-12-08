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


def get_train_data(width, num_rewards, episode_length, horizon, cost, test=False):
    data_path = f'/scratch1/fs1/chien-ju.ho/RIS/518/scripts/{width}_{num_rewards}_{episode_length}_{horizon}.npz'
    all_data = np.load(data_path)
    x, y = all_data['x'], all_data['y']
    y = (y>cost).astype(int)
    pos, neg = (y==1), (y==0)
    train_n, eval_n = 80000, 20000
    n = train_n + eval_n
    xpos, xneg = x[pos][:n], x[neg][:n]
    ypos, yneg = y[pos][:n], y[neg][:n]
    
    if test:
        xeval = np.concatenate([xpos[-eval_n:],xneg[-eval_n:]])
        yeval = np.concatenate([ypos[-eval_n:],yneg[-eval_n:]])
        return xeval, yeval
        
    xtrain = np.concatenate([xpos[:train_n],xneg[:train_n]])
    ytrain = np.concatenate([ypos[:train_n],yneg[:train_n]])
    return xtrain, ytrain

for width in [6]:
    for num_rewards in [4]:
        for episode_length in [5,6,7,8]:
            for cost in [0,5,10]:
                for train_horizon in [1, 2, 3]:
                    xtrain, ytrain = get_train_data(width, num_rewards, episode_length, train_horizon, cost, test=False)
                    tfp = get_tfp(filters=64, blocks=6, regularizer=False, input_size=3, board_size=width, output_size=1)
                    optimizer = tfp.optimizer
                    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                    metrics = ['accuracy',tf.keras.metrics.AUC()]
                    tfp.model.compile(optimizer, loss, metrics)

                    tfp.model.fit(xtrain,ytrain, verbose=0)
                    for test_horizon in [1, 2, 3]:
                        xeval, yeval = get_train_data(width, num_rewards, episode_length, test_horizon, cost, test=True)
                        res = tfp.model.evaluate(xeval,yeval, verbose=0)
                        print(f'{episode_length}\t{cost}\t{train_horizon}\t{test_horizon}\t{res[1]}')
                    print()
                print('-----')