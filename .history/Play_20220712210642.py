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
net.load_state_dict(checkpoint["policy_state_dict"])

def epsilon_greedy(state, epsilon):
        assert epsilon <= 1 and epsilon >= 0, "Epsilon needs to be in the range of [0, 1]"
        # take random action -> exploration 
        rand = np.random.random()

        if rand <= epsilon:
            return np.random.randint(0, num_actions-1)
        else:
            with torch.no_grad():
                self.policy.eval()
                # normalize the state array then make it compatible for pytorch 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                net_out = self.policy(state)
            # take action based on the highest Q value -> exploitation 
            return int(net_out.argmax())

# play tthe game 
for episode in range(3):
    state = wrapped_env.reset()
    done = False

    for step in range(max_episode_steps):
        # as you can see, epsilon is decreasing for each episode 
        action = epsilon_greedy(state, epsilon=0.05)
        next_state, reward, done, info = wrapped_env.step(action)

        # set the next state to None if the game is over
        if done:
            next_state = None 
            break

              






