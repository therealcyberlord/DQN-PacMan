import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device


torch.load("Checkpoints/mspacmanNet-episode-500.chkpt", device=device)
env = gym.make("MsPacmanNoFrameskip-v4")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
wrapped_env = FrameStack(wrapped_env, 4)

