import torch
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from configs.config import cfg, args
from dataset.utils import CLASSES


def save_model(net, optim, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}__'.format(cfg.network) + '{}.pth'.format(epoch)))


def load_model(network, model_dir):
    if not os.path.exists(model_dir):
        return False
    pths = [pth.split('.')[0] for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        return False
    pths = [int(pth.split('__')[1]) for pth in pths]
    if cfg.test_epoch == -1:
        pth = max(pths)
    else:
        pth = cfg.test_epoch
    model = torch.load(os.path.join(model_dir, '{}__'.format(cfg.network) + '{}.pth'.format(pth)))
    network.load_state_dict(model['net'], strict=True)
    return True


def time2hour(seconds):
    hour = int(seconds / 3600)
    minute = int((seconds - hour * 3600) / 60)
    second = int(seconds - hour * 3600 - minute * 60)
    return hour, minute, second


def unnorm(image, mean, std):
    return np.clip(image * std + mean, 0, 1)


def plot_line(data, title, mode):
    x = range(len(data))
    if mode == 'iters':
        multi = 20
    else:
        multi = 1
    x = [int(x_ * multi) for x_ in x]
    plt.figure("{}".format(title))
    plt.plot(x, data, 'b-', label=title)
    plt.title("{}".format(title))
    plt.xlabel('{}'.format(mode))
    plt.ylabel('{}'.format(title))
    plt.legend()
    plt.ylim(ymin=0)
    save_dir = os.path.join(cfg.result_dir, title)
    plt.savefig(save_dir)


def save_result():
    os.system('mkdir -p {}'.format(cfg.result_dir))
    plot_line(cfg.train_loss, 'train_loss', 'iters')
    plot_line(cfg.val_loss, 'val_loss', 'epoch')
    plot_line(cfg.train_acc, 'train_acc', 'epoch')
    plot_line(cfg.val_acc, 'val_acc', 'epoch')
    if args.show:
        plt.show()


def visuzlization(network, data_loader):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    network.eval()
    for data in tqdm(data_loader):
        images, labels = data
        outputs = network(images.cuda())
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            print('GroundTruth: {}'.format(CLASSES[labels[i]]))
            print('Predicted: {}'.format(CLASSES[predicted[i]]))

            # vis
            image = images[i].permute(1, 2, 0).cpu().numpy()
            image = unnorm(image, mean, std)
            plt.imshow(image)
            plt.title("GT: {}\n".format(CLASSES[labels[i]]) + "Pred: {}".format(CLASSES[predicted[i]]))
            plt.show()
