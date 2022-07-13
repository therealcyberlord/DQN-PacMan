class Logger:
    def __init__(self):
        self.episodes = []
        self.steps = [] 
        self.rewards = []

    def print_stats(self, episode_num, episode_rewards, episode_steps, total_steps, record=True):
        print("Episode #:", episode_num)
        print("Rewards:", episode_rewards)
        print("Steps:", episode_steps)
        print("Total Steps:", total_steps)
        print("--------------------")
        print("")

        if record:
            self.episodes.append(episode_num)
            self.steps.append(episode_steps)
            self.rewards.append(episode_rewards)

    def summary_stats(self, num_episodes):
        pass
