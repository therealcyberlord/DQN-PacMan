import torch
from torch import nn 
import torch.nn.functional as F
from Configs import device

# This will be the neural network that we are training on

class DQN(nn.Module):
    def __init__(self, output_dim):
        super(DQN, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(64)

        # (84, 84, 1) -> (40, 40, 16) -> (18, 18, 32) -> (7, 7, 64)

        self.fc1 = nn.Linear(7*7*64, output_dim)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        return F.relu(self.fc1(x))

if __name__ == "__main__":
    dqn = DQN(output_dim=5)
