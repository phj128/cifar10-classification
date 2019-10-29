import torch
from tqdm import tqdm
from dataset.utils import CLASSES


def cal_acc(net, data_loader, mode):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the {}'.format(total) +
          ' {} '.format(mode) + 'images: %f %%' % (100.0 * correct / total))
    return 100.0 * correct / total


def cal_acc_cat(net, data_loader, mode):
    class_correct = [0. for _ in range(10)]
    class_total = [0. for _ in range(10)]
    with torch.no_grad():
        for data in tqdm(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print("\nAccuracy in {}:\n".format(mode))

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (CLASSES[i], 100 * class_correct[i] / class_total[i]))

