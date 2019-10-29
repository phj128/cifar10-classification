import torch.nn as nn
import torch.nn.functional as F


class fcnet(nn.Module):
    def __init__(self):
        super(fcnet, self).__init__()
        self.iter = 3
        self.pre = nn.Linear(32 * 32 * 3, 128)
        self.hide = []
        for i in range(self.iter):
            fc = nn.Linear(2 ** i * 128, 2 ** (i + 1) * 128)
            self.__setattr__('fc' + str(i), fc)
        for i in range(self.iter):
            fc = nn.Linear(2 ** (self.iter - i) * 128, 2 ** (self.iter - i - 1) * 128)
            self.__setattr__('fc' + str(i + self.iter), fc)
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.pre(x))
        for i in range(2 * self.iter):
            fc = self.__getattr__('fc'+str(i))
            x = F.relu(fc(x))
        x = self.out(x)
        return x


def get_fc8():
    return fcnet()
