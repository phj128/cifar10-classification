import torch
import torchvision
import torchvision.transforms as transforms
from configs.config import cfg


def make_dataset(mode='net'):
    if mode == 'vis':
        # For visuzlizaion
        transform_test = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        test_set = torchvision.datasets.CIFAR10(root=cfg.data_dirs, train=False,
                                                download=True, transform=transform_test)

        test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                                  shuffle=False, num_workers=cfg.num_workers)
        return test_loader

    """
    For training:
    padding 4 and random crop to get 32*32
    p=0.5 for ramdom flip
    angle（-5,5） random rotation
    normalize the picture
    """
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(5, resample=False, expand=False, center=None),
         transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_set = torchvision.datasets.CIFAR10(root=cfg.data_dirs, train=True,
                                             download=True, transform=transform_train)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size,
                                               shuffle=True, num_workers=cfg.num_workers)
    """
    For test:
    Do nothing but normalize
    """
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    test_set = torchvision.datasets.CIFAR10(root=cfg.data_dirs, train=False,
                                            download=True, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=cfg.batch_size,
                                              shuffle=False, num_workers=cfg.num_workers)

    return train_loader, test_loader

