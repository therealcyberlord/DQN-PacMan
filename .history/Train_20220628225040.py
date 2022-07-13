import torch
import numpy as np 

class Train_DQN:
    def __init__(self, env, height, width, channel, device, memory, policy, target, gamma, optim):
        self.env = env 
        self.height = height
        self.width = width
        self.channel = channel
        self.device = device 
        self.memory = memory
        self.policy = policy 
        self.target = target
        self.gamma = gamma 
        self.optim = optim


    def epsilon_greedy(self, epsilon):
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


    def optimize_model(self, policy, target, replay_buffer, gamma, optim, criterion, batch_size):
        # sample from the replay buffer
        batch = replay_buffer.sample(batch_size)
        
        # converting data from the batch to be compatible with pytorch, also normalize the image data
        
        states = np.array([s[0] for s in batch]).astype("float32") / 255.0
        states = torch.tensor(states, dtype=torch.float32, device=self.device).view(-1, 1, self.height, self.width)

        actions = torch.tensor(np.array([s[1] for s in batch]), dtype=torch.int64, device = self.device)
        rewards = torch.tensor(np.array([s[3] for s in batch]), dtype=torch.float32, device = self.device)

        non_final_next_states = np.array([s[2] for s in batch if s[2] is not None]).astype("float32") / 255.0
        non_final_next_states = torch.tensor(non_final_next_states, dtype=torch.float32, device = self.device).view(-1, 1, self.height, self.width)

        non_final_mask = torch.tensor(np.array([s[2] is not None for s in batch]), dtype=torch.bool)

        # computing the Q values 

        policy.train()
        q_values = policy(states)
        
        # Select the proper Q value for the corresponding action taken Q(s_t, a)
        state_action_values = q_values.gather(1, actions.unsqueeze(1).cuda())

        # Compute the value function of the next states using the target network 
        with torch.no_grad():
            target.eval()
            q_values_target = target(non_final_next_states)
            next_state_max_q_values = torch.zeros(batch_size, device = self.device)
            next_state_max_q_values[non_final_mask] = q_values_target.max(dim=1)[0].detach()

        # update based on value iteration 
        expected_state_action_values = rewards + (next_state_max_q_values * gamma)
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # Compute the Huber loss
        loss = criterion(state_action_values, expected_state_action_values)

        # Optimize the model
        optim.zero_grad()
        loss.backward()
        optim.step()
    

    def learn(self, num_episodes, max_episode_steps):
        for episode in range(num_episodes):
            state = self.env.reset()

            episode_reward = 0 
            episode_steps = 0 

            done = False

            for step in range(max_episode_steps):
                # as you can see, epsilon is decreasing for each episode 
                action = self.epsilon_greedy(wrapped_env, policy_net, state, max(1 - episode / 500, end_epsilion_decay))
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
            
            

