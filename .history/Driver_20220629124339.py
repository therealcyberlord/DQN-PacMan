# will collect experiences based on the policy 
from tqdm import tdqm 

class DynamicStepDriver:
    def __init__(self, env, replay_buffer):
        self.env = env 
        self.replay_buffer = replay_buffer
        state = self.env.reset()

    def collect(self, num_steps):
        print("Collecting experiences")
        done = False 
        for step in tqdm(range(num_steps)):
            action = self.env.action_space.sample()
            state, next_state, reward, info = self.env.step(action)

            if done:
                next_state = None
                state = self.env.reset()
    
            state = next_state

            self.replay_buffer.push((state, action, next_state, reward))
    
        

