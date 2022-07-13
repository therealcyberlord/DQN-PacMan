class Logger:
    def __init__(self):
        self.steps = [] 
        self.rewards = []

    def print_stats(self, episode_num, episode_rewards, episode_steps, total_steps):
        print("Episode #:", episode_num)
        print("Rewards:", episode_rewards)
        print("Steps:", episode_steps)
        print("Total Steps:", total_steps)
        print("--------------------")
        print("")

    def summary_stats(num_episodes):
        pass
