# help us keep track of the results during training 

class Logger:
    def __init__(self):
        self.episodes = []
        self.steps = [] 
        self.rewards = []
        self.total_steps = [] 

    def print_stats(self):
        # this prints out the latest stats
        print("Episode #:", self.episodes[-1])
        print("Rewards:", self.steps[-1])
        print("Steps:", self.rewards[-1])
        print("Total Steps:", self.total_steps[-1])
        print("--------------------")
        print("")

    def record(self, episode_num, episode_rewards, episode_steps, total_steps):
        self.episodes.append(episode_num)
        self.steps.append(episode_steps)
        self.rewards.append(episode_rewards)
        self.total_steps.append(total_steps)

    def summary_stats(self, num_episodes):
        pass
