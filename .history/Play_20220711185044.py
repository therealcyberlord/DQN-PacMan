import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device, max_episode_steps
import numpy as np 

checkpoint = torch.load("Checkpoints/mspacmanNet-episode-500.chkpt", map_location=device)

env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
wrapped_env = FrameStack(wrapped_env, 4)
num_actions = wrapped_env.action_space.n


# we'll be using two networks for training the Atari AI
net = DQN(num_actions, device).to(device)
net.load_state_dict(checkpoint["policy_state_dict"])


for i in range(3):
    state = wrapped_env.reset()
    done = False 
    for step in range(max_episode_steps):
        with torch.no_grad():
            net.eval()
            # normalize the state array then make it compatible for pytorch 
            state = np.array(state, dtype=np.float32) / 255.0
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(1)
            net_out = net(state)
            
        action = int(net_out.argmax())
        next_state, reward, done, info = wrapped_env.step(action)
        if done:
            break
        state = next_state
        










