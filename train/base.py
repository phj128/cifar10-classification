import torch
from tqdm import tqdm
from configs.config import cfg


def train_base(network, criterion, data_loader, optimizer, epoch, mode='train'):
    if mode == 'train':
        running_loss = 0.0
        network.train()
        for i, data in enumerate(data_loader, 0):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if (i + 1) % 20 == 0:
                print('Epoch%d: iter: %d, train_loss: %.3f' % (epoch + 1, i+1, running_loss / 20))
                cfg.train_loss.append(running_loss / 20)
                running_loss = 0.0
    else:
        with torch.no_grad():
            network.eval()
            running_loss = 0.0
            i = 0
            for data in tqdm(data_loader):
                i += 1
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                outputs = network(inputs)
                loss = criterion(outputs, labels)

                # print statistics
                running_loss += loss.item()

        print('Epoch%d: val_loss: %.3f' % (epoch + 1, running_loss / i))
        cfg.val_loss.append(running_loss / i)


