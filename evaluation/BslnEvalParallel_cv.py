import gymnasium as gym
import numpy as np
from torch import nn as nn
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

#from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage, stacked_observations, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym

tmp_path = "/home/asalvi/code_workspace/tmp/sb3_log/VisServo/test/"

ref_wp = '/home/asalvi/code_workspace/Husky_CS_SB3/Paths/QueryPoints/'



#for X in ['exp0','exp1','exp2','exp3','exp4']

#for X in ['Vel0p75','Vel0p6','Vel0p45','Vel0p3','Vel0p15']:
#for X in ['2Wheel0p75','2Wheel0p6','2Wheel0p45','2Wheel0p3','2Wheel0p15']:

X = '216'
Y = '0.15'


def make_env(env_id, rank, X, Y, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        port_no = str(23004 + 2*rank)
        print(port_no)
        seed = 1 + rank
        #trt_ = throttle_map.get(trt, 20)
        trt_ = 20
        track_vel_map = {
        '0.75': 0.75,
        '0.6': 0.6,
        '0.45': 0.45,
        '0.3': 0.3,
        '0.15': 0.15
        }
        track_vel = track_vel_map.get(Y, 0.75)
        
        track_vel = 0.75
        env = gym.make(env_id, port=port_no, seed=1 + rank, track_vel=track_vel)
        #env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init

if __name__ == '__main__':
    env_id = "huskyCP_gym/HuskyRL-v0"
    num_cpu = 1  # Number of processes to use
    # Create the vectorized environment
    n_eval_episodes = 11

    for X in ['Bsln']:
    #for X in ['216','288','432','864','2160']:
        for Y in ['0.15']:
            env = SubprocVecEnv([make_env(env_id, i, X, Y) for i in range(num_cpu)], start_method='fork')
            rank = 0
            port_no = str(23004 + 2*rank)
            print(port_no)
            seed = 1 + rank
            #trt_ = throttle_map.get(trt, 20)
            trt_ = 20
            track_vel_map = {
            '0.75': 0.75,
            '0.6': 0.6,
            '0.45': 0.45,
            '0.3': 0.3,
            '0.15': 0.15
            }
            track_vel = track_vel_map.get(Y, 0.75)
            
            #track_vel = 0.75
            #env = gym.make(env_id, port=port_no, seed=1 + rank, track_vel=track_vel, speci= X + '_' + Y, pth = X,n_eval_episodes = n_eval_episodes)
            #env = VecMonitor(env, filename=tmp_path)
            #env = VecTransposeImage(env, skip=False)
            #env = VecNormalize(env, training=True, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99, epsilon=1e-08, norm_obs_keys=None)


            # Create environment
            #env = gym.make("huskyCP_gym/HuskyRL-v0",port=23004,seed=1,track_vel = 0.75)

            model_path = f'/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/eval_policies/IKModel/2Wheel0p15'
    
            #model_path = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/guided/bslnCnst'
            #model_path = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/Bsln/bslns2/bslnCnst.zip'

            model = PPO.load(model_path, env=env, print_system_info=True)

            mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes, deterministic = True)
            obs = env.reset()
            print(f'{X}_{Y}policy Complete')