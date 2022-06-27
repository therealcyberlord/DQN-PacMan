from collections import deque
import random 

# the replay buffer stores experiences, which the agent learns from 

class ReplayBuffer(object):

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def push(self, *args):
        self.buffer.append(*args)

    def sample(self, batch_size):
        assert batch_size <= len(self), "Batch size cannot be greater than the size of the deque"
        return random.sample(self.buffer, batch_size)
        
    def __len__(self):
        return len(self.buffer)