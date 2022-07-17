# will collect experiences based on the policy 
from tqdm import tqdm
import random
from Configs import num_actions


class DynamicStepDriver:
    def __init__(self, env, replay_buffer):
        self.env = env 
        self.replay_buffer = replay_buffer

    # collect data for the replay buffer 
    def collect(self, num_steps):
        print("Collecting experiences")
        done = False 
        state = self.env.reset()

        
        for step in tqdm(range(num_steps)):
            # random action
            action = random.randint(0, num_actions-1)
          
            # stepping into the environment 
            next_state, reward, done, info = self.env.step(action)
            self.replay_buffer.push((state, action, next_state, reward))
            state = next_state 

            if done:
                state = self.env.reset()
    
        

