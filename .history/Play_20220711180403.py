import torch
import gym 
from Model import DQN
from Configs import device

torch.load("Checkpoints/mspacmanNet-episode-500.chkpt", device=device)
