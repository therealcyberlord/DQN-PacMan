import torch
import gym 
from gym.wrappers import AtariPreprocessing, RecordVideo
from Model import DQN
from Configs import device, max_episode_steps, num_actions
import numpy as np 
import argparse 
import PIL
import os 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-episodes", help="number of episodes to play", default=3, type=int)
    parser.add_argument("-checkpoint", help="which checkpoint to restore to", default=1200)
    parser.add_argument("-record", help="whether to record the gameplay", action='store_true', default=False)

    args = parser.parse_args()

    checkpoint = torch.load(f"Checkpoints/mspacmanNet-episode-{args.checkpoint}.chkpt", map_location=device)

    env = gym.make("MsPacmanNoFrameskip-v4")

    # apply the standard atari preprocessing -> convert to grayscale, frameskip, resize to 84x84
    wrapped_env = AtariPreprocessing(env)

    # record the environment
    if args.record:
        wrapped_env = RecordVideo(wrapped_env, video_folder="Gameplay", name_prefix="mspacman-gameplay")

    net = DQN(num_actions).to(device)

    # load checkpoint data for the policy model
    net.load_state_dict(checkpoint["policy_state_dict"])

    for i in range(args.episodes):
        state = wrapped_env.reset()
        done = False 
        for step in range(max_episode_steps):
            with torch.no_grad():
                net.eval()
                # normalize the state array then make it compatible for the trained dqn 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                net_out = net(state)
        
            action = torch.argmax(net_out, dim=1).item()
            next_state, reward, done, info = wrapped_env.step(action)

            if done:
                break
            # update the current state since we moved
            state = next_state

    
if __name__ == "__main__":
    main()










