import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device, max_episode_steps, width, height, num_actions
import numpy as np 

checkpoint = torch.load("Checkpoints/mspacmanNet-episode-1200.chkpt", map_location=device)

env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)

# we'll be using two networks for training the Atari AI
net = DQN(num_actions).to(device)
# net.load_state_dict(checkpoint["policy_state_dict"])


for episode in range(3):
    state = wrapped_env.reset()
    done = False

    for step in range(max_episode_steps):
        # as you can see, epsilon is decreasing for each episode 
        action = self.epsilon_greedy(state, epsilon=0.05)
        next_state, reward, done, info = wrapped_env.step(action)

        # set the next state to None if the game is over
        if done:
            next_state = None 
            break

              






