from functools import reduce

from torch import nn
from torch.nn import functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, reduce(lambda x, y: x * y, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn3 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.bn4 = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.bn5 = nn.BatchNorm1d(84)
        self.drop = nn.Dropout()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bn3(x)
        x = x.view(-1, reduce(lambda x, y: x * y, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = self.bn4(x)
        x = F.relu(self.fc2(x))
        x = self.bn5(x)
        x = self.drop(x)
        x = self.fc3(x)
        return x


class RichCNN(nn.Module):
    def __init__(self):
        super(RichCNN, self).__init__()
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 1024)
        self.bn4 = nn.BatchNorm1d(1024)
        self.drop = nn.Dropout()
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.bn1(x)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = self.bn2(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = self.bn3(x)
        x = x.view(-1, reduce(lambda x, y: x * y, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class PracticalCNN(nn.Module):
    def __init__(self):
        super(PracticalCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, 5)
        self.conv2 = nn.Conv2d(5, 50, 5)
        self.fc1 = nn.Linear(50 * 5 * 5, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, reduce(lambda x, y: x * y, x.size()[1:]))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FullyConnected(nn.Module):
    def __init__(self):
        super(FullyConnected, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
