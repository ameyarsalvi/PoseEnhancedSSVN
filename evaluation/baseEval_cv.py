import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym

class GetEnvVar(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        #self.training_env = env

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value0 =  self.training_env.get_attr("self.lin_Vel")
        #value0 = self.locals_['self.lin_Vel']
        #value = self.get_attr('self.log_err_feat')
        #self.logger.record("random_value", value)
        print(value0)
        #return value
   

# Create environment
env = gym.make("huskyCP_gym/HuskyRL-v0",port=23004,seed=1,track_vel = 0.75)

#model_path = '/home/asalvi/Downloads/WP150.zip'
model_path = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/Bsln/bslns2/bslnCnst.zip'

model = PPO.load(model_path, env=env, print_system_info=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25, deterministic = True)
obs = env.reset()