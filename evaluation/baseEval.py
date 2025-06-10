## /home/asalvi/Downloads/bslnPEVN.zip

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
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage, stacked_observations, VecMonitor

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym")
import huskyCP_gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=208):
        super().__init__(observation_space, features_dim)

        # Image processing CNN
        n_input_channels = observation_space["image"].shape[2]
        self.cnn = nn.Sequential(
          nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
          nn.ReLU(),
          nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
          nn.ReLU(),
          nn.Flatten()
      )

        # Calculate CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space["image"].sample()[None]).permute(0, 3, 1, 2).float()
            ).shape[1]

        '''
        try:
            dummy_input = torch.as_tensor(observation_space["image"].sample()[None]).permute(0, 3, 1, 2).float()
            print("Dummy input shape before passing to CNN:", dummy_input.shape)
            n_flatten = self.cnn(dummy_input).shape[1]
            print("CNN output shape:", self.cnn(dummy_input).shape)
        except Exception as e:
            print("Error during CNN initialization:", str(e))
            raise
        '''


        # Fully connected layer for vector input
        self.fc = nn.Sequential(
            nn.Linear(observation_space["vector"].shape[0],8),
            nn.ReLU()
        )

        # Combine CNN and vector outputs
        self.fc_combined = nn.Sequential(
            nn.Linear(n_flatten + 8, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
      # Move tensors to the appropriate device (GPU or CPU)
      device = next(self.cnn.parameters()).device  # Get the device of the model
      #print("Image shape before permute:", observations["image"].shape)
      image = observations["image"].to(device).permute(0, 3, 1, 2)  # Move image to device and permute
      #print("Image shape after permute:", image.shape)
      vector = observations["vector"].to(device)  # Move vector to device

      # Process image through CNN
      cnn_out = self.cnn(image)

      # Process vector through fully connected layer
      vector_out = self.fc(vector)

      # Concatenate and process combined features
      combined = torch.cat([cnn_out, vector_out], dim=1)
      return self.fc_combined(combined)

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

def make_env(env_id, rank, seed=0):
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
        #env = gym.make("huskyCP_gym/HuskyRL-v0",port=23004,seed=1,track_vel = 0.75, paths_dir = "/home/asalvi/code_workspace/Husky_CS_SB3/PoseEnhancedVN/train/MixPathFlip/",
        #       variant = 'bslnPEVN',log_option = 0, eval_dir = "/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/Policies/expTrt/expDump/expDumpAll/")
        env = gym.make(env_id,port=23004,seed=1,track_vel = 0.75)
        #env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init
   

# Create environment
#env = 

#tmp_path = "/home/asalvi/code_workspace/tmp/RedRes2/2WE/Eval" # Path to save logs
# Create environment
#env = gym.make("huskyCP_gym/HuskyRL-v0",port=23004,seed=1,track_vel = 0.75,log_option = 0)
env_id = "huskyCP_gym/HuskyRL-v0"
num_cpu = 1  # Number of processes to use
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
#env = VecMonitor(env, filename=tmp_path)
env = VecTransposeImage(env, skip=False)
#env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99, epsilon=1e-08, norm_obs_keys=None)

model_path = '/home/asalvi/Downloads/ImgCent.zip'
#model_path = '/home/asalvi/code_workspace/Husky_CS_SB3/Evaluation/EvalDump/Bsln/bslns2/bslnCnst.zip'

model = PPO.load(model_path, env=env, print_system_info=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25, deterministic = True)
obs = env.reset()