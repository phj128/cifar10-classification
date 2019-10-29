import torch.nn as nn


def make_loss():
    criterion = nn.CrossEntropyLoss()
    return criterion

