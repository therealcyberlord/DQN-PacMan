import torch 

# variables we need for setting up the agent 
max_episode_steps = 27000 
gamma = 0.99
replay_mem_size = 100000
batch_size = 64
lr = 0.005
min_samples_for_training = 20000
update_target_net_steps = 5000
num_episodes = 1000 
end_epsilion_decay = 0.05
height = width = 84
channel = 1
save_checkpoint_period = 100

# we are limiting the agent to either move left, right, up, down, and no action
num_actions = 5

# If there is a CUDA-compatible GPU, use it for training, otherwise use the CPU
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")



