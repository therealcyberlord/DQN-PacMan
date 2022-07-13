# will collect experiences based on the policy 

class DynamicStepDriver:
    def __init__(self, env, replay_buffer):
        self.env = env 
        self.replay_buffer = replay_buffer
        self.env.reset()

    def collect(self, num_steps):
        print("Collecting experiences")
        done = False 
        for step in num_steps:
            action = self.env.action_space.sample()
            state, next_state, reward, info = self.env.step(action)

            if done:
                next_state = None

            self.replay_buffer.push((state, action, next_state, reward))

