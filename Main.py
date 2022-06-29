import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import RMSprop
from torch.nn import HuberLoss
import torch
from Replay import ReplayBuffer
from Train import Train_DQN
from Model import DQN
from Driver import DynamicStepDriver


# variables we need for setting up the agent 
max_episode_steps = 27000 
gamma = 0.99
replay_mem_size = 100000
batch_size = 64
lr = 0.005
min_samples_for_training = 10000
train_period = 4
update_target_net_steps = 5000
num_episodes = 500 
end_epsilion_decay = 0.10
height = width = 84
channel = 1
memory = ReplayBuffer(replay_mem_size)


# check if there is a CUDA compatible GPU, otherwise use the CPU 
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")

# initialize the atari environment 
env = gym.make("MsPacmanNoFrameskip-v4")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
wrapped_env = FrameStack(wrapped_env, 4)

num_actions = wrapped_env.action_space.n

# we'll be using two networks for training the Atari AI
policy_net = DQN(num_actions, device).to(device)
target_net = DQN(num_actions, device).to(device)

# defining the optimizer and criterion 
optimizer = RMSprop(policy_net.parameters(), lr=lr)
criterion = HuberLoss().to(device)

# initalize the target network with the same weights, we'll periodically update this 
# it will be used for evaluation, so the weights won't be adjusted
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# collecting experiences using a random policy -> replay buffer
driver = DynamicStepDriver(wrapped_env, memory)
driver.collect(min_samples_for_training)

print(len(memory))

# create the agent which will learn to play the environment 
Agent = Train_DQN(wrapped_env, height, width, batch_size, channel, device, memory, policy_net, target_net, gamma, optimizer, criterion)
Agent.learn(num_episodes, max_episode_steps, min_samples_for_training, train_period, update_target_net_steps, end_epsilion_decay)






