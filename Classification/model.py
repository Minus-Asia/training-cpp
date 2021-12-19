import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # the input has size [4, 3, 32, 32] batch size = 4, 3 channel image, width = 32, heigh = 32
        # print("input size:", x.size())
        # self.conv1 takes input 3 channels and
        # and output 6 channels, kernel size = 5, the output size will be computed by
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # after conv2d: [4, 6, 28, 28]
        # after relu:  [4, 6, 28, 28]
        # after pool: [4, 6, 14, 14]
        x = self.pool(F.relu(self.conv1(x)))
        # print("1 output size:", x.size())
        # output has size [4, 16, 5, 5]
        x = self.pool(F.relu(self.conv2(x)))
        # print("2 output size:", x.size())
        # output has size [4, 400] after flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # print("3 output size:", x.size())
        # Just reduce dimention from 400 to 120
        x = F.relu(self.fc1(x))
        # print("4 output size:", x.size())
        # Just reduce dimention from 120 to 84
        x = F.relu(self.fc2(x))
        # print("5 output size:", x.size())
        # Just reduce dimention from 84 to 10 due to number of cifa-10 classes = 10
        x = self.fc3(x)
        # print("-------------------------------------------")
        return x
