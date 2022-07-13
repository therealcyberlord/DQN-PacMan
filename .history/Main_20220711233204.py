import gym
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import RMSprop
from torch.nn import HuberLoss
import torch
from Replay import ReplayBuffer
from Train import Train_DQN
from Model import DQN
from Configs import *
from Driver import DynamicStepDriver
import argparse 


def main():
    # command line arguments for training 
    
    parser = argparse.ArgumentParser()

    parser.add_argument("-episodes", help="number of episodes to train on", default=num_episodes, type=int)
    parser.add_argument("-lr", help="learning rate", default=lr, type=float)
    parser.add_argument("-replay_size", help="size of the replay buffer", default=replay_mem_size, type=int)
    parser.add_argument("-batch_size", help="batch size", default=batch_size, type=int)
    parser.add_argument("-gamma", help="discount rate", default=gamma, type=float)
    parser.add_argument("-end_epsilion_decay", help="when to stop decaying epsilon", default=end_epsilion_decay, type=float)
    parser.add_argument("--update_target_steps", help="when to update the target network with the policy network", default=update_target_net_steps, type=int)
    parser.add_argument("-max_episode_steps", help="max steps in a given episode before termination", default=max_episode_steps, type=int)
    parser.add_argument("-save_checkpoint", help="save checkpoint every given episodes", default=save_checkpoint_period, type=int)



    args = parser.parse_args()

    # set up the replay buffer
    memory = ReplayBuffer(args.replay_size)

    # initialize the atari environment 
    env = gym.make("MsPacmanNoFrameskip-v4")

    # apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
    wrapped_env = AtariPreprocessing(env)

    num_actions = wrapped_env.action_space.n

    # we'll be using two networks for training the Atari AI
    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)

    # defining the optimizer and criterion 
    optimizer = RMSprop(policy_net.parameters(), lr=args.lr)
    criterion = HuberLoss().to(device)

    # initalize the target network with the same weights, we'll periodically update this 
    # it will be used for evaluation, so the weights won't be adjusted
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # collecting experiences using a random policy -> replay buffer
    driver = DynamicStepDriver(wrapped_env, memory)
    driver.collect(min_samples_for_training)

    # create the agent which will learn to play the environment 
    Agent = Train_DQN(wrapped_env, height, width, args.batch_size, channel, device, memory, policy_net, target_net, args.gamma, optimizer, criterion, args.save_checkpoint)
    Agent.learn(args.episodes, args.max_episode_steps, update_target_net_steps, args.end_epsilion_decay)


if __name__ == "__main__":
    main()


