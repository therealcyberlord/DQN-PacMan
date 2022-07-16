import numpy as np 
import matplotlib.pyplot as plt 
# help us keep track of the progress during training

class Logger:
    def __init__(self, num_episodes, average_reward_period):
        self.num_episodes = num_episodes 
        self.episodes = []
        self.steps = [] 
        self.rewards = []
        self.average_rewards = [] 
        self.total_steps = [] 
        self.average_period = average_reward_period
        self.best_average = 0 

    def print_stats(self):
        # this prints out the latest stats
        assert self.episodes is not None, "please record before printing"

        print(f"Episode {self.episodes[-1]}/{self.num_episodes}")
        print("Rewards:", self.rewards[-1])
        print("Average Rewards: %.4f" % self.average_rewards[-1])
        print("Best Average Rewards: %.4f" % self.best_average)
        print("Steps:", self.steps[-1])
        print("Total Steps:", self.total_steps[-1])
        print("--------------------")
        print("")

    # record statistics after an episode
    def record(self, episode_num, episode_rewards, episode_steps, total_steps):
        self.episodes.append(episode_num)
        self.rewards.append(episode_rewards)
        self.average_rewards.append(np.mean(self.rewards[-self.average_period:]))
        self.steps.append(episode_steps)
        self.total_steps.append(total_steps)

    # check if the current average is better than the best average 
    def new_best_average(self):
        if self.average_rewards[-1] > self.best_average:
            self.best_average = self.average_rewards[-1]
            return True 
        return False 


    def plot(self):
        # plot the average reward over the episodes
        plt.figure(figsize=(8, 6))
        plt.title("Rewards over the episodes")
        plt.plot(self.average_rewards)
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.show()

        # plot the huber loss over the episodes
        plt.figure(figsize=(8, 6))
        plt.title("Loss over the episodes")
        plt.plot(self.huber_loss)
        plt.xlabel("Episodes")
        plt.ylabel("Huber Loss")
        plt.show()
