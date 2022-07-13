import torch
from torch import nn 
import torch.nn.functional as F

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

        self.fc1 = nn.Linear(4*7*7*64, 32)
        self.fc2 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # flattening the tensor for fc layers
        x = x.view(-1, 4*7*7*64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


