import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(44944, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 90)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Net(nn.Module):
    def __init__(self, filter_1, filter_2, filter_3, kernel_1, kernel_2, kernel_3):
        super().__init__()
        # print(filter_1, "filter_1")
        # print(filter_2, "filter_2")
        # print(filter_3, "filter_3")
        # print(kernel_1, "kernel_1")
        # print(kernel_2, "kernel_2")
        # print(kernel_3, "kernel_3")
        self.conv1 = nn.Conv2d(3, filter_1, kernel_size=(kernel_1, kernel_1))
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(
            filter_1, filter_2, kernel_size=(kernel_2, kernel_2))
        self.conv3 = nn.Conv2d(
            filter_2, filter_3, kernel_size=(kernel_3, kernel_3))

        self.fc1 = nn.LazyLinear(90)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # print(x.shape, "shape")
        x = self.pool(F.relu(self.conv2(x)))
        # print(x.shape, "shape")
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)

        x = self.fc1(x)

        return x
