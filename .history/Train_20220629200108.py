import torch
import numpy as np 
from Logging import Logger

class Train_DQN:
    def __init__(self, env, height, width, batch_size, channel, device, replay_buffer, policy, target, gamma, optim, criterion, save_period):
        self.env = env 
        self.height = height
        self.width = width
        self.batch_size = batch_size
        self.channel = channel
        self.device = device 
        self.replay_buffer = replay_buffer
        self.policy = policy 
        self.target = target
        self.gamma = gamma 
        self.optim = optim
        self.criterion = criterion
        self.save_period = save_period 


    def epsilon_greedy(self, state, epsilon):
        assert epsilon <= 1 and epsilon >= 0, "Epsilon needs to be in the range of [0, 1]"
        # take random action -> exploration 
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                self.policy.eval()
                # normalize the state array then make it compatible for pytorch 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32).view(-1, self.channel, self.height, self.width)
                net_out = self.policy(state)
            # take action based on the highest Q value -> exploitation 
            return int(net_out.argmax())


    def optimize_model(self):
        # sample from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # converting data from the batch to be compatible with pytorch, also normalize the image data
        
        states = np.array([experience[0] for experience in batch]).astype("float32") / 255.0
        # shape (256, 1, 84, 84)
        states = torch.tensor(states, dtype=torch.float32, device=self.device).view(-1, 1, self.height, self.width)
        actions = torch.tensor(np.array([experience[1] for experience in batch]), dtype=torch.int64, device = self.device)
        rewards = torch.tensor(np.array([experience[3] for experience in batch]), dtype=torch.float32, device = self.device)

        non_final_next_states = np.array([experience[2] for experience in batch if experience[2] is not None]).astype("float32") / 255.0
        non_final_next_states = torch.tensor(non_final_next_states, dtype=torch.float32, device = self.device).view(-1, 1, self.height, self.width)

        non_final_mask = torch.tensor(np.array([experience[2] is not None for experience in batch]), dtype=torch.bool)

        # computing the Q values 

        self.policy.train()
        q_values = self.policy(states)
        
        # Select the proper Q value for the corresponding action taken Q(s_t, a)
        state_action_values = q_values.gather(1, actions.unsqueeze(1))

        # Compute the value function of the next states using the target network 
        with torch.no_grad():
            self.target.eval()
            q_values_target = self.target(non_final_next_states)
            next_state_max_q_values = torch.zeros(self.batch_size, device = self.device)
            next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

        # update based on value iteration 
        expected_state_action_values = rewards + (next_state_max_q_values * self.gamma)
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # Compute the Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
    

    def learn(self, num_episodes, max_episode_steps, min_samples_for_training, train_period, update_target_net_steps, end_epsilion_decay):
        step_counter = 0
        # initialize the logger
        logger = Logger()

        for episode in range(num_episodes):
            state = self.env.reset()

            episode_reward = 0 
            episode_steps = 0 

            done = False

            for step in range(max_episode_steps):
                # as you can see, epsilon is decreasing for each episode 
                action = self.epsilon_greedy(state, max(1 - episode / 500, end_epsilion_decay))
                next_state, reward, done, info = self.env.step(action)
                # incrementing onto the episode reward
                episode_reward += reward

                step_counter += 1
                episode_steps += 1

                # set the next state to None if the game is over
                if done:
                    next_state = None 
                    break

                # we add the experiences to the replay buffer
                self.replay_buffer.push((state, action, next_state, reward))

                # train the model every several steps 
                if step_counter % train_period == 0:
                    self.optimize_model()

                # update the target network
                if step_counter % update_target_net_steps == 0:
                    print("Update target network")
                    self.target.load_state_dict(self.policy.state_dict())

                # set current state to next state
                state = next_state
                
            # printing out episode stats
            logger.record(episode+1, episode_reward, episode_steps, step_counter) 
            logger.print_stats()
        # plot the rewards function
        logger.plot()
            
            

