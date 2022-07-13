class Logger:
    def __init__(self):
        pass

    def print_stats(self, episode_num, episode_rewards, episode_steps, total_steps):
        print("Episode #:", episode_num)
        print("Rewards:", episode_rewards)
        print("Steps:", episode_steps)
        print("Total Steps:", total_steps)
        print("--------------------")

