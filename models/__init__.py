from __future__ import absolute_import

from models.PSTA import PSTA

__factory = {
    'PSTA' : PSTA,
}

def get_names():
    return __factory.keys()

def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)
# import torch
# a=torch.randn(8,128,3)
# for idx in range(0, 3, 3):
#     # print(idx)
#     # print(channel_para[:, :, idx].shape,channel_para[:, :, idx + 1].shape)
#     para0 = torch.cat((a[:, :, idx], a[:, :, idx + 1]), 1)
#     print(para0.shape)