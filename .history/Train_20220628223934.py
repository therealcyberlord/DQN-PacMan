import torch
import numpy as np 

class Train_DQN:
    def __init__(self, height, width, channel, device):
        self.height = height
        self.width = width
        self.channel = channel
        self.device = device 

    def epsilon_greedy(self, env, net, state, epsilon):
        assert epsilon <= 1 and epsilon >= 0, "Epsilon needs to be in the range of [0, 1]"
        # take random action -> exploration 
        if np.random.random() <= epsilon:
            return env.action_space.sample()
        else:
            with torch.no_grad():
                net.eval()
                # normalize the state array then make it compatible for pytorch 
                state = np.array(state, dtype=np.float32) / 255.0
                state = torch.tensor(state, dtype=torch.float32).view(-1, self.channel, self.height, self.width)
                net_out = net(state)
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
            next_state_max_q_values = torch.zeros(batch_size, device=device)
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
    
    

