import numpy as np 
import matplotlib.pyplot as plt 
# help us keep track of the progress during training

class Logger:
    def __init__(self, num_episodes):
        self.num_episodes = num_episodes 
        self.episodes = []
        self.steps = [] 
        self.rewards = []
        self.total_steps = [] 
        self.huber_loss = []

    def print_stats(self):
        # this prints out the latest stats
        assert self.episodes is not None, "please record before printing"

        print(f"Episode {self.episodes[-1]}/{self.num_episodes}")
        print("Rewards:", self.rewards[-1])
        print("Loss:", self.huber_loss[-1])
        print("Steps:", self.steps[-1])
        print("Total Steps:", self.total_steps[-1])
        print("--------------------")
        print("")

    # record statistics after an episode
    def record(self, episode_num, episode_rewards, loss, episode_steps, total_steps):
        self.episodes.append(episode_num)
        self.rewards.append(episode_rewards)
        self.steps.append(episode_steps)
        self.huber_loss.append(loss)
        self.total_steps.append(total_steps)

    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.title("Rewards over the episodes")
        average = np.mean(self.rewards)
        plt.plot(self.rewards)
        plt.axhline(y = average, color = 'r', linestyle = 'dashed')
        plt.xlabel("Episodes")
        plt.ylabel("Scores")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.title("Loss over the episodes")
        average = np.mean(self.huber_loss)
        plt.plot(self.huber_loss)
        plt.axhline(y = average, color = 'r', linestyle = 'dashed')
        plt.xlabel("Episodes")
        plt.ylabel("Huber Loss")
        plt.show()
