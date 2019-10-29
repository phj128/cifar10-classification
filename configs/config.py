from yacs.config import CfgNode as CN
import argparse
import os


cfg = CN()

cfg.model = 'debug'

cfg.network = 'res_18'
cfg.gpu = '0'
cfg.batch_size = 2
cfg.num_workers = 8
cfg.epoch = 60

cfg.weight_decay = 5e-4
cfg.lr = 0.01
cfg.warm_up = 2
cfg.warm_lr = 1e-4
cfg.milestones = [30, 50]
cfg.gamma = 0.1

cfg.work_dirs = './work_dirs'
cfg.data_dirs = './data'

cfg.train_loss = []
cfg.val_loss = []
cfg.train_acc = []
cfg.val_acc = []

cfg.test_epoch = -1
cfg.test_train = False

def make_cfg(args):
    cfg.merge_from_list(args.opts)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    cfg.model_dir = os.path.join(cfg.work_dirs, cfg.model)
    cfg.result_dir = os.path.join(cfg.model_dir, 'result')
    return cfg


parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', dest='test', default=False)
parser.add_argument('--show', action='store_true', dest='show', default=False)
parser.add_argument('--vis', action='store_true', dest='vis', default=False)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
args = parser.parse_args()
cfg = make_cfg(args)
