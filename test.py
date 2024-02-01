import csv
import sys
import torchvision
sys.setrecursionlimit(15000)
import os
import torch
import numpy as np
from torch.autograd import Variable
import torch.utils.data
import argparse
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
from models import PSTA_8_att
from utils import accuracy_score1,DeepSupervision_acc2
from sklearn.metrics import accuracy_score
from res2net_v1b import res2net101_v1b
import l2cs1
import attributecnn


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
parser.add_argument('--outf', default='/media/gpu/Elements/bc/model', help='folder to output model checkpoints')
parser.add_argument('--random', action='store_true', default=False, help='enable randomness for routing matrix')
parser.add_argument('--id', type=int, default=0, help='checkpoint ID')

opt = parser.parse_args()
print(opt)


class MyDatasetff(Dataset):
    def __init__(self, root_dir, clip_len, transforms_=None, test_sample_num=1, stride=8):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.stride = stride
        self.class2idx = {'fakeff': 0, 'realff': 1}
        self.class_count = [0] * 2
        self.fake_count = 0

        for base, subdirs, files in os.walk(self.root_dir):
            if len(files) < self.stride * self.clip_len:  #
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.jpg'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            data['video'] = video
            data['label'] = 0 if 'fakeff' in base else 1
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]['video']
        label = self.data[idx]['label']
        length = len(video)
        clip_start =0
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]
        a = clip[1]['frame']
        cnames = a.rsplit('/', 1)[0]
        cname = cnames.rsplit('_', 1)[0]
        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = Image.open(frame['frame'])
                frame = self.transforms_(frame)  # tensor [C x H x W]

                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        else:
            clip = torch.tensor(clip)
        return clip, torch.tensor(int(label)),cname


class MyDatasetwild(Dataset):

    def __init__(self, root_dir, clip_len, transforms_=None, test_sample_num=1, stride=8):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.transforms_ = transforms_
        self.test_sample_num = test_sample_num
        self.toPIL = transforms.ToPILImage()
        self.data = []
        self.stride = stride
        self.class2idx = {'fake': 0, 'real': 1}
        self.class_count = [0] * 2
        self.fake_count = 0

        for base, subdirs, files in os.walk(self.root_dir):
            if len(files) < self.stride * self.clip_len:  #
                continue
            data = {}
            video = []
            files.sort()
            for i, f in enumerate(files):
                if f.endswith('.png'):
                    data_dict = {}
                    data_dict['frame'] = os.path.join(base, f)
                    data_dict['index'] = i
                    video.append(data_dict)
            data['video'] = video
            data['class'] = self.class2idx[base.split('/')[-2]]
            self.class_count[data['class']] += 1
            data['label'] = 0 if 'fake_' in base else 1
            self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            clip (tensor): [channel x time x height x width]
            class_idx (tensor): class index, [0-100]
        """
        video = self.data[idx]['video']
        label = self.data[idx]['label']
        sub_class = self.data[idx]['class']
        length = len(video)
        clip_start=0
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]
        a = clip[1]['frame']
        cnames = a.rsplit('/', 1)[0]
        cname = cnames.rsplit('_', 1)[0]
        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            seed = random.random()
            for frame in clip:
                random.seed(seed)
                frame = Image.open(frame['frame'])
                frame = self.transforms_(frame)  # tensor [C x H x W]

                trans_clip.append(frame)
            # (T x C X H x W) to (C X T x H x W)
            clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        else:
            clip = torch.tensor(clip)
        return clip, torch.tensor(int(label)),cname




if __name__ == '__main__':

    text_writer = open(os.path.join(opt.outf, 'test.txt'), 'w')

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
    ])
    testset = MyDatasetff('.../your testdataset',4,
                         test_transforms)

    print(len(testset))
    test_bs =8
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)

    ##############################################
    model1 = l2cs1.L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90).cuda(opt.gpu_id)
    model1.load_state_dict(torch.load(os.path.join(opt.outf, 'model1_' + str(opt.id) + '.pt')))
    model1.eval()
    
    model2 = l2cs1.L2CS(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 90).cuda(opt.gpu_id)
    model2.load_state_dict(
        torch.load('/data/Capsule-Forensics-v2-master (1)/gazemodel/L2CSNet_gaze360.pkl'))
    for i in model2.parameters():
        i.requires_grad = False
    model2.eval()

   

    modelcnn = res2net101_v1b()
    modelcnn.load_state_dict(torch.load(os.path.join(opt.outf, 'modelcnn_' + str(opt.id) + '.pt')))
    modelcnn.eval()


    modelatt=attributecnn.ResNet34().cuda(opt.gpu_id)
    modelatt.load_state_dict(torch.load(os.path.join(opt.outf, 'modelatt_' + str(opt.id) + '.pt')))
    modelatt.eval()

    modelatt1=attributecnn.ResNet34().cuda(opt.gpu_id)
    for i in modelatt1.parameters():
        i.requires_grad = False
    modelatt1.eval()



    capnet = PSTA_8_att.PSTA(num_classes=2, seq_len=12).cuda(opt.gpu_id)
    capnet.load_state_dict(torch.load(os.path.join(opt.outf, 'capsule_' + str(opt.id) + '.pt')))
    capnet.eval()

    if opt.gpu_id >= 0:
        model1.cuda(opt.gpu_id)
        model2.cuda(opt.gpu_id)
        modelcnn.cuda(opt.gpu_id)
        modelatt.cuda(opt.gpu_id)
        modelatt1.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)
    ##############################################



    tol_label = np.array([], dtype=np.float)
    tol_pred = np.array([], dtype=np.float)
    tol_pred_prob = np.array([], dtype=np.float)

    count = 0
    loss_test = 0
    acc_test = 0
    cname=[]
    for batch_idx, test_data0 in enumerate(testloader, 1):

        test_data, test_target,img_name = test_data0
        data = Variable(test_data)
        target = Variable(test_target)
        target[target > 1] = 1
        img_label = target.numpy().astype(float)
        data, target = data.cuda(opt.gpu_id), target.cuda(opt.gpu_id)
        img_data = data.cuda(opt.gpu_id)
        labels_data = target.cuda(opt.gpu_id)

        img_data = img_data.permute(0, 2, 1, 3, 4)
        b, t, c, w, h = img_data.size()
        img_datas = img_data.contiguous().view(b * t, c, w, h)

        gaze, gaze_bias, feat_map = model1(img_datas)  # (b * t, c, 16, 8)
        gaze2, gaze_bias2, feat_map2 = model2(img_datas)
        cnnout,prediction1 = modelcnn(img_datas)
        cnnout1=modelatt(img_datas)
        outputs, features = capnet(img_data, feat_map, cnnout,cnnout1)


        if isinstance(outputs, (tuple, list)):
            acc_te,mylabel,confidence_fake,confidence_real = DeepSupervision_acc2(accuracy_score, outputs, labels_data,features)

        else:
            acc_te = accuracy_score1(outputs, labels_data).cuda(opt.gpu_id)

        acc_test += acc_te

        count += 1

        # 将输入数据转换为二维列表

        data = [[fp, cf,cr,mlabel] for fp, cf,cr,mlabel in zip(img_name, confidence_fake,confidence_real,mylabel)]
        # 写入CSV文件
      
        labelcsv='/data/Capsule-Forensics-v2-master (1)/experiment/resnet/notextureloss/trainff_testcele/40.csv'
        with open(labelcsv, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)


    acc_test /= count
    print('[Epoch %d] Test acc: %.2f' % (opt.id, acc_test * 100))
    text_writer.write('%d,%.2f\n' % (opt.id, acc_test * 100))

    text_writer.flush()
    text_writer.close()

