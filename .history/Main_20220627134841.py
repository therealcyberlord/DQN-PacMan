import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import RMSprop
from torch.nn import HuberLoss
from Replay import ReplayBuffer
from Train import epsilon_greedy, optimize_model
from Model import device, DQN


# variables we need for setting up the agent 
max_episode_steps = 27000 
gamma = 0.99
replay_mem_size = 100000
batch_size = 64
lr = 2.5e-4
min_samples_for_training = 30000
train_period = 4
update_target_net_steps = 5000
num_episodes = 500 
end_epsilion_decay = 0.10

# count the number of steps 
step_counter = 0

memory = ReplayBuffer(replay_mem_size)

# initialize the atari environment 
env = gym.make("MsPacmanNoFrameskip")

# apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
wrapped_env = AtariPreprocessing(env)
wrapped_env = FrameStack(wrapped_env, 4)

num_actions = wrapped_env.action_space.n

# we'll be using two networks for training the Atari AI
policy_net = DQN(num_actions).to(device)
target_net = DQN(num_actions).to(device)

# defining the optimizer and criterion 
optimizer = RMSprop(policy_net.parameters(), lr=lr)
criterion = HuberLoss()

# initalize the target network with the same weights, we'll periodically update this 
# it will be used for evaluation, so the weights won't be adjusted
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


for episode in range(num_episodes):
    state = wrapped_env.reset()

    episode_reward = 0 
    episode_steps = 0 

    done = False

    for step in range(max_episode_steps):
        # as you can see, epsilon is decreasing for each episode 
        action = epsilon_greedy(wrapped_env, policy_net, state, max(1 - episode / 500, end_epsilion_decay))
        next_state, reward, done, info = wrapped_env.step(action)
        # incrementing onto the episode reward
        episode_reward += reward

        step_counter += 1
        episode_steps += 1

        # set the next state to None if the game is over
        if done:
            next_state = None 
            break

        # we add the experiences to the replay buffer
        memory.push((state, action, next_state, reward))

        # once the memory hits the minimum crtieria, we start training once every four steps
        if len(memory) > min_samples_for_training:
            if step % train_period == 0:
                optimize_model(policy_net, target_net, memory, gamma, optimizer, criterion, batch_size)

            # update the target network
            if step_counter % update_target_net_steps == 0:
                print("Update target network")
                target_net.load_state_dict(policy_net.state_dict())

        # set current state to next state
        state = next_state
        
    # printing out episode stats
    print(f"episode: {episode + 1} - episode reward: {episode_reward}  episode steps: {episode_steps}  total steps: {step_counter}\n") 
    




