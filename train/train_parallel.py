
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
from torch.optim import Adam

from stable_baselines3.common.logger import configure

from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage, stacked_observations, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed


from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch


###### Extract var arguments from json ##########
import json

with open("args.json","r") as f:
    config = json.load(f)

args = config["args"][0]

env_registration_dir = args['env_registration_dir']
paths_dir = args['paths_dir']
variant = args['variant']
tensor_logs = args['tensor_logs']
log_option = args['log_options']
eval_dir = args['eval_dir']

#################################################

import sys
sys.path.insert(0, env_registration_dir) #Ensure correct path
import huskyCP_gym

tmp_path = tensor_logs # Path to save logs
#variant = 'test' # Save final model by this name

# Create log dir
import os
# set up logger
os.makedirs(tmp_path, exist_ok=True)
new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

total_timesteps = 2e7


# Callback Definitions

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

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              self.logger.record('mean_reward', mean_reward)
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True



def make_env(env_id, 
             rank, 
             paths_dir, 
             variant, 
             log_option, 
             eval_dir):
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
        env = gym.make(env_id, 
                       port = port_no,
                       seed = seed, 
                       track_vel = 0.75, 
                       paths_dir = paths_dir, 
                       variant = variant, 
                       log_option = log_option, 
                       eval_dir = eval_dir)
        #env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init



if __name__ == '__main__':
    env_id = "huskyCP_gym/HuskyRL-v0"
    num_cpu = 4  # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i,paths_dir, variant, log_option, eval_dir) for i in range(num_cpu)], start_method='fork')
    env = VecMonitor(env, filename=tmp_path)
    env = VecTransposeImage(env, skip=False)
    env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99, epsilon=1e-08, norm_obs_keys=None)


    best_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=tmp_path)
    checkpoint_callback = CheckpointCallback(save_freq=78125,save_path=tmp_path + "checkpoints/",name_prefix=variant,save_replay_buffer=True,save_vecnormalize=True)
    callback = CallbackList([best_callback,checkpoint_callback])

'''
    model = PPO("CnnPolicy", env,learning_rate=0.0001, n_steps=512, batch_size=512, n_epochs=5, ent_coef= 0.005, gamma=0.98, gae_lambda=0.98,
                clip_range=0.1, vf_coef=0.5, max_grad_norm=0.5,sde_sample_freq=16, 
                policy_kwargs=dict(normalize_images=True, log_std_init=-1.0,ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[64], vf=[64])), 
                verbose=1, tensorboard_log=tmp_path)
                
                '''
model = PPO(
    "MultiInputPolicy", 
    env,
    learning_rate=0.00001, 
    n_steps=512, 
    batch_size=512, 
    n_epochs=5, 
    ent_coef=0.005, 
    gamma=0.98, 
    gae_lambda=0.98,
    clip_range=0.1, 
    vf_coef=0.5, 
    max_grad_norm=0.5, 
    sde_sample_freq=16, 
    policy_kwargs=dict(
        features_extractor_class=CustomCombinedExtractor,
        features_extractor_kwargs=dict(features_dim=208),
        normalize_images=True,
        log_std_init=-1.0,
        ortho_init=False,
        activation_fn=nn.ReLU,
        net_arch=dict(pi=[64], vf=[64]),
        optimizer_class=Adam,  # Use Adam optimizer
        optimizer_kwargs=dict(weight_decay=1e-4)  # Add L2 regularization here
    ),
    verbose=1, 
    tensorboard_log=tmp_path
)
    
model.set_logger(new_logger)
model.learn(total_timesteps, callback=callback, progress_bar= True)
model.save(tmp_path + variant)

obs = env.reset()
    #for _ in range(1000):
    #    action, _states = model.predict(obs)
    #    obs, rewards, dones, info = env.step(action)
