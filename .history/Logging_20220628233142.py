class Logging:
    def __init__(self, episode_num, episode_rewards, episode_steps, total_steps):
        self.episode_num = episode_num
        self.epsiode_rewards = episode_rewards
        self.episode_steps = episode_steps
        self.total_steps = total_steps 

    def print_stats(self):
        print("Episode #:", self.episode_num)
        print("Rewards:", self.epsiode_rewards)
        print("Steps:", self.episode_steps)
        print("Total Steps:", self.total_steps)

