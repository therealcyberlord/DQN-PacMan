import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device, max_episode_steps, width, height
import random
import numpy as np 

checkpoint = torch.load("Checkpoints/mspacmanNet-episode-500.chkpt", map_location=device)

env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
num_actions = wrapped_env.action_space.n


# we'll be using two networks for training the Atari AI
net = DQN(num_actions).to(device)
# net.load_state_dict(checkpoint["policy_state_dict"])


for i in range(3):
    state = wrapped_env.reset()
    done = False 
    for step in range(max_episode_steps):
        if random.random() <= 0.30:
            action = random.randint(0, 4)
        else:
            with torch.no_grad():
                net.eval()
                # normalize the state array then make it compatible for pytorch 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).view(-1, 1, height, width)
                net_out = net(state)
            # take action based on the highest Q value -> exploitation 
            action = int(net_out.argmax())

        next_state, reward, done, info = wrapped_env.step(action)

        if done:
            break
        # update the current state since we moved
        state = next_state
        










