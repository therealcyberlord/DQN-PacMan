# will collect experiences based on the policy 
from tqdm import tqdm


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
            action = self.env.action_space.sample()
          
            # stepping into the environment 
            next_state, reward, done, info = self.env.step(action)

            if done:
                next_state = None
                state = self.env.reset()

            self.replay_buffer.push((state, action, next_state, reward))

            # if next_state is not none, we would like to continue in the environment 
            if next_state is not None:
                state = next_state
    
        

