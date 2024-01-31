from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp
import numpy as np
import torch
from sklearn.metrics import f1_score,accuracy_score
# from .euclidean_loss import EuclideanLoss
from collections import Counter

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

def disciminative(x):
    above_average = x >= np.min(x)
    r = np.zeros(above_average.shape[0])
    r[above_average] = 1 / np.sum(above_average)
    return r

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AttributesMeter(object):
    """Computes and stores the average and current value.

       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self, attr_num):
        self.attr_num = attr_num
        self.preds =  [[] for _ in range(attr_num)]
        self.gts = [[] for _ in range(attr_num)]
        self.acces = np.array([0 for _ in range(attr_num)])
        self.acces_avg = None
        self.f1_score_macros = None
        self.count = 0

    def update(self, preds, gts, acces, n):
        self.count += n
        self.acces += acces
        for i in range(len(preds)):
            self.preds[i].append(preds[i])
            self.gts[i].append(gts[i])

    def get_f1_and_acc(self, mean_indexes=None):
        if mean_indexes is None:
            mean_indexes = [_ for _ in range(self.attr_num)]
        if self.acces_avg is None:
            self.acces_avg = self.acces / self.count
        if self.f1_score_macros is None:
            self.f1_score_macros = np.array([f1_score(y_pred=self.preds[i], y_true=self.gts[i], average='macro') for i in [0, 1] + list(range(self.attr_num))])

        return self.f1_score_macros, self.acces_avg, np.mean(self.acces_avg[mean_indexes]), np.mean(self.f1_score_macros[mean_indexes])



def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_model.pth.tar'))

class Logger(object):
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))



def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    return optimizer


def make_optimizer_with_center(cfg, model, center_criterion):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)
    return optimizer, optimizer_center

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

def DeepSupervision(criterion , xs, y):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """
    loss = 0.
    for x in xs:
        loss += criterion(x, y)
    loss /= len(xs)
    # print(loss)
    return loss


def find_most_frequent(nums):


    result = []
    for col in range(len(nums[0])):
        if len(nums) != 3 or len(nums[0]) != len(nums[1]) or len(nums[0]) != len(nums[2]):
        #if len(nums) != 4 :#zi
            continue
        else:
            cnt = Counter([num[col] for num in nums])
            most_common = cnt.most_common(1)
            result.append(most_common[0][0])
    return result


def find_most_frequent_8_32(nums):


    result = []
    for col in range(len(nums[0])):
        if len(nums) != 4 or len(nums[0]) != len(nums[1]) or len(nums[0]) != len(nums[2]):
        #if len(nums) != 4 :#zi
            continue
        else:
            cnt = Counter([num[col] for num in nums])
            most_common = cnt.most_common(1)
            result.append(most_common[0][0])
    return result


def find_most_frequent3(nums):


    result = []
    for col in range(len(nums[0])):
        if len(nums) != 2 or len(nums[0]) != len(nums[1]):
        #if len(nums) != 4 :#zi
            continue
        else:
            cnt = Counter([num[col] for num in nums])
            most_common = cnt.most_common(1)
            result.append(most_common[0][0])
    return result






def DeepSupervision_acc_32_2(accuracy_score , xs, y):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """


    #y=y.detach().cpu().numpy()
    acc = 0.
    for x in xs:
        #x = x.detach().cpu().numpy()
        _, predicted_labels = torch.max(x, 1)
        # print(predicted_labels,'orilabel')
        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    return acc


def DeepSupervision_acc(accuracy_score , xs, y):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """


    #y=y.detach().cpu().numpy()
    acc = 0.
    for x in xs:
        #x = x.detach().cpu().numpy()
        _, predicted_labels = torch.max(x, 1)
        # print(predicted_labels,'orilabel')
        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    return acc


def DeepSupervision_acc1(accuracy_score , xs, y,features):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """
    confidence_fake=[]
    confidence_real= []

    tensor1=features[0]
    tensor2=features[1]
    tensor3=features[2]


    #tensor4=features[3]#zi
    #4层
    # result = []
    # for t1, t2, t3,t4 in zip(tensor1, tensor2, tensor3,tensor4):
    #     t = (t1 + t2 + t3+t4) / 4
    #     result.append(t)

    result = []
    for t1, t2, t3 in zip(tensor1, tensor2, tensor3):
        t = (t1 + t2 + t3) / 3
        result.append(t)

    for j in result:
        fake_c = j[0].item()
        real_c=j[1].item()
        confidence_fake.append(fake_c)
        confidence_real.append(real_c)





    acc = 0.
    nums=[]
    for x in xs:
        _, predicted_labels = torch.max(x, 1)

        xlist=predicted_labels.tolist()
        nums=nums+xlist

        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    small_lists = [nums[i:i + 8] for i in range(0, len(nums),8)]#zi batchsize
    #small_lists = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    mylabel=find_most_frequent(small_lists)




    return acc,mylabel,confidence_fake,confidence_real

def accuracy_score1( xs, y):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """
    # xs = xs.detach().cpu().numpy()
    # y = y.detach().cpu().numpy()
    acc=0
    _, predicted_labels = torch.max(xs, 1)
    acc+= accuracy_score(predicted_labels.numpy() , y.numpy())

    return acc




def DeepSupervision_acc2(accuracy_score , xs, y,features):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """
    confidence_fake=[]
    confidence_real= []

    tensor1=features[0]
    tensor2=features[1]
    tensor3=features[2]


    result = []
    for t1, t2, t3 in zip(tensor1, tensor2, tensor3):
        t = (t1 + t2 + t3) / 3
        result.append(t)

    for j in result:
        fake_c = j[0].item()
        real_c=j[1].item()
        confidence_fake.append(fake_c)
        confidence_real.append(real_c)



    batchsize=8

    acc = 0.
    nums=[]
    for x in xs:
        _, predicted_labels = torch.max(x, 1)

        xlist=predicted_labels.tolist()
        nums=nums+xlist

        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    small_lists = [nums[i:i + batchsize] for i in range(0, len(nums),batchsize)]#zi batchsize
    # print(small_lists,'small')
    #small_lists = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    mylabel=find_most_frequent(small_lists)
    # print(mylabel,'mylabel')




    return acc,mylabel,confidence_fake,confidence_real




def DeepSupervision_acc_8_32(accuracy_score , xs, y,features):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """

    confidence_fake=[]
    confidence_real= []

    tensor1=features[0]
    tensor2=features[1]
    tensor3 = features[2]
    tensor4 = features[3]

    result = []
    for t1, t2,t3,t4 in zip(tensor1, tensor2, tensor3, tensor4):
        t = (t1 + t2 +t3+t4) / 4
        result.append(t)

    for j in result:
        fake_c = j[0].item()
        real_c=j[1].item()
        confidence_fake.append(fake_c)
        confidence_real.append(real_c)



    batchsize=2

    acc = 0.
    nums=[]
    for x in xs:
        _, predicted_labels = torch.max(x, 1)

        xlist=predicted_labels.tolist()
        nums=nums+xlist

        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    small_lists = [nums[i:i + batchsize] for i in range(0, len(nums),batchsize)]#zi batchsize
    # print(small_lists,'small')
    #small_lists = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    mylabel=find_most_frequent_8_32(small_lists)
    # print(mylabel,'mylabel')




    return acc,mylabel,confidence_fake,confidence_real




def DeepSupervision_acc3(accuracy_score , xs, y,features):
    """DeepSupervision

        Applies criterion to each element in a list.

        Args:
            criterion: loss function
            xs: tuple of inputs
            y: ground truth
        """
    confidence_fake=[]
    confidence_real= []

    tensor1=features[0]
    tensor2=features[1]
    # tensor3=features[2]
    # print(tensor1.shape,tensor2.shape)

    result = []
    for t1, t2 in zip(tensor1, tensor2):
        t = (t1 + t2 ) / 2
        result.append(t)

    for j in result:
        fake_c = j[0].item()
        real_c=j[1].item()
        confidence_fake.append(fake_c)
        confidence_real.append(real_c)





    acc = 0.
    nums=[]
    for x in xs:
        _, predicted_labels = torch.max(x, 1)

        xlist=predicted_labels.tolist()
        nums=nums+xlist

        acc+= accuracy_score(predicted_labels.detach().cpu().numpy() , y.detach().cpu().numpy())
    acc /= len(xs)
    small_lists = [nums[i:i + 8] for i in range(0, len(nums),8)]#zi batchsize
    # print(small_lists,'small')
    #small_lists = [nums[i:i + 4] for i in range(0, len(nums), 4)]
    mylabel=find_most_frequent3(small_lists)
    # print(mylabel,'mylabel')



    # print(acc,mylabel,confidence_fake,confidence_real)
    return acc,mylabel,confidence_fake,confidence_real