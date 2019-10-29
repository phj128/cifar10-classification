import torch.nn as nn
import torch.nn.functional as F


class fcnet(nn.Module):
    def __init__(self):
        super(fcnet, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.fc2 = nn.Linear(512, 2048)
        self.fc3 = nn.Linear(2048, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_fc3():
    return fcnet()
