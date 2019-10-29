import torch.optim as optim
from configs.config import cfg


def make_optim(net, lr):
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=cfg.weight_decay)
    return optimizer
