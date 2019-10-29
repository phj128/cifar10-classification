from configs.config import cfg, args
from dataset.cifar10 import make_dataset
from network.network_factory import make_network
from train.optim import make_optim
from train.loss import make_loss
from train.base import train_base
from utils.utils import time2hour, save_result, save_model, load_model, visuzlization
from vallidation.acc import cal_acc, cal_acc_cat
import time


def train():
    print("Start init...")
    train_loader, test_loader = make_dataset()
    model = make_network()
    network = model()
    criterion = make_loss()
    network.cuda()
    optimizer = make_optim(network, cfg.warm_lr)

    print("Start training...")
    for epoch in range(cfg.epoch):
        start_time = time.time()

        if epoch + 1 == cfg.warm_up:
            optimizer = make_optim(network, cfg.lr)

        if (epoch + 1) in cfg.milestones:
            cfg.lr /= 10
            optimizer = make_optim(network, cfg.lr)

        train_base(network, criterion, train_loader, optimizer, epoch)

        end_time = time.time()
        used_time = end_time - start_time
        _, u_minute, u_second = time2hour(used_time)
        l_hour, l_minute, l_second = time2hour(used_time * (cfg.epoch - epoch - 1))
        print("Finish one epoch in %dm: %ds, and %dh: %dm: %ds left." % (u_minute, u_second, l_hour, l_minute, l_second))
        train_base(network, criterion, test_loader, optimizer, epoch, mode='test')

        train_acc = cal_acc(network, train_loader, 'train')
        test_acc = cal_acc(network, test_loader, 'test')
        cfg.train_acc.append(train_acc)
        cfg.val_acc.append(test_acc)

        if (epoch + 1) % 10 == 0:
            save_model(network, optimizer, epoch+1, cfg.model_dir)

    save_model(network, optimizer, cfg.epoch, cfg.model_dir)

    print('Finish Training')
    save_result()


def test():
    print("Start init...")
    train_loader, test_loader = make_dataset()
    model = make_network()
    network = model()
    flag = load_model(network, cfg.model_dir)
    if not flag:
        print("The model doesn't exist!!!")
        raise AssertionError
    network.cuda()
    if args.vis:
        vis_loader = make_dataset(mode='vis')
        visuzlization(network, vis_loader)
    mode = 'test'
    loader = test_loader
    if cfg.test_train:
        mode = 'train'
        loader = train_loader
    cal_acc(network, loader, mode)
    cal_acc_cat(network, loader, mode)
    print("Finish all")


def main():
    if args.test:
        test()
    else:
        train()


if __name__ == "__main__":
    main()
