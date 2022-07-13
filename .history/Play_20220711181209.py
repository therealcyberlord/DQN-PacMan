import torch
import gym 
from gym.wrappers import AtariPreprocessing, FrameStack
from Model import DQN
from Configs import device, max_episode_steps
import numpy as np 

torch.load("Checkpoints/mspacmanNet-episode-500.chkpt", device=device)
dqn_policy = DQN()

env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
wrapped_env = FrameStack(wrapped_env, 4)


# we'll be using two networks for training the Atari AI
policy_net = DQN(wrapped_env.action_space.n).to(device)

for i in range(10):
    state = wrapped_env.reset()
    for step in range(max_episode_steps):
         with torch.no_grad():
                policy_net.eval()
                # normalize the state array then make it compatible for pytorch 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(1)
                net_out = policy_net(state)
                action = int(net_out.argmax())
                next_state, reward, done, info = env.step(action)









