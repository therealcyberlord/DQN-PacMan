# will collect experiences based on the policy 

class DynamicStepDriver:
    def __init__(self, env):
        self.env = env 
        self.env.reset()

    def collect(self, num_steps):
        for step in num_steps:
            action = self.env.action_space.sample()
            state, next_state, reward, info = self.env.step(action)
            
