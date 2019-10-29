from configs.config import cfg
from .fc_3 import get_fc3
from .fc_6 import get_fc6
from .fc_8 import get_fc8
from .fc_10 import get_fc10
from .res_18 import get_res18
from .res_34 import get_res34
from .res_50 import get_res50
from .res_101 import get_res101


res_factory = {
    '18': get_res18,
    '34': get_res34,
    '50': get_res50,
    '101': get_res101,
}


fc_factory = {
    '3': get_fc3,
    '6': get_fc6,
    '8': get_fc8,
    '10': get_fc10,
}


def network_factory(net, num):
    if net == 'res':
        return res_factory[num]
    else:
        return fc_factory[num]


def make_network():
    net_type = cfg.network.split('_')[0]
    net_num = cfg.network.split('_')[1]
    return network_factory(net_type, net_num)
