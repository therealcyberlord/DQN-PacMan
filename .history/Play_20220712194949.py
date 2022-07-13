import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device, max_episode_steps, width, height, num_actions
import numpy as np 

checkpoint = torch.load("Checkpoints/mspacmanNet-episode-200.chkpt", map_location=device)

env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)

# we'll be using two networks for training the Atari AI
net = DQN(num_actions).to(device)
# net.load_state_dict(checkpoint["policy_state_dict"])


for i in range(3):
    state = wrapped_env.reset()
    done = False 
    for step in range(max_episode_steps):
        with torch.no_grad():
            net.eval()
            # normalize the state array then make it compatible for pytorch 
            state = np.array(state, dtype=np.float32) / 255.0
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            net_out = net(state)
        # take action based on the highest Q value -> exploitation 
        action = int(net_out.argmax())

        next_state, reward, done, info = wrapped_env.step(action)

        if done:
            break
        # update the current state since we moved
        state = next_state
        










