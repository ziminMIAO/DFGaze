import sys
sys.setrecursionlimit(15000)
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim import Adam,SGD
import torch.utils.data
import argparse
import l2cs1
import torchvision
from models import PSTA_8,PSTA_8_att
from utils import DeepSupervision,DeepSupervision_acc,accuracy_score1

from sklearn.metrics import accuracy_score
import os

import numpy as np
import attributecnn
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from res2net_v1b import res2net101_v1b




parser = argparse.ArgumentParser()
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam')
parser.add_argument('--gpu_id', type=int, default=1, help='GPU ID')
parser.add_argument('--resume', type=int, default=0, help="choose a epochs to resume from (0 to train from scratch)")
parser.add_argument('--outf', default='/media/gpu/Elements/bc/model', help='folder to output model checkpoints')
parser.add_argument('--disable_random', action='store_true', default=False, help='disable randomness for routing matrix')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout percentage')
opt = parser.parse_args()
print(opt)

opt.random = not opt.disable_random



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
        self.class_count = [0] *2
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
      

        clip_start =0
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]
        a = clip[1]['frame']
        b=clip[2]['frame']
       
        cnames = a.rsplit('/', 1)[0]
        cname = cnames.rsplit('_', 1)[0]

        if self.transforms_:
            trans_clip = []
            # fix seed, apply the sample `random transformation` for all frames in the clip
            #seed = random.random()
            for frame in clip:
                #random.seed(seed)
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
   
        video = self.data[idx]['video']
        label = self.data[idx]['label']
        sub_class = self.data[idx]['class']
        length = len(video)
        clip_start=0
        clip = video[clip_start: clip_start + (self.clip_len * self.stride): self.stride]
  
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


        return clip, torch.tensor(int(label))



if __name__ == "__main__":

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if opt.gpu_id >= 0:
        torch.cuda.manual_seed_all(opt.manualSeed)
        cudnn.benchmark = True

    if opt.resume > 0:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'a')
    else:
        text_writer = open(os.path.join(opt.outf, 'train.csv'), 'w')



    model1 = l2cs1.L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90).cuda(opt.gpu_id)
    model1.load_state_dict(
    torch.load('/data/Capsule-Forensics-v2-master (1)/gazemodel/L2CSNet_gaze360.pkl'))

    model2 = l2cs1.L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], 90).cuda(opt.gpu_id)
    model2.load_state_dict(
    torch.load('/data/Capsule-Forensics-v2-master (1)/gazemodel/L2CSNet_gaze360.pkl'))
    for i in model2.parameters():
        i.requires_grad = False


    capnet=PSTA_8_att.PSTA(num_classes=2, seq_len=12).cuda(opt.gpu_id)


    modelcnn=res2net101_v1b()
    model_path ='/data/Capsule-Forensics-v2-master (1)/res2net model/premodel_res2net.pth'
    modelcnn.load_state_dict(torch.load(model_path))



    modelatt1 = attributecnn.ResNet34() 
    for i in modelatt1.parameters():
        i.requires_grad = False

    modelatt = attributecnn.ResNet34() 


    print('--------------model1--------------------')
    print(model1)
    print('--------------modelcnn--------------------')
    print(modelcnn)
    print('----------------modelatt------------------')
    print(modelatt)
    print('----------------capnet------------------')
    print(capnet)


    capsule_loss = torch.nn.CrossEntropyLoss().cuda(opt.gpu_id)
    l1loss=torch.nn.L1Loss().cuda(opt.gpu_id)


    xent = torch.nn.CrossEntropyLoss().cuda(opt.gpu_id)
    tent = torch.nn.CrossEntropyLoss().cuda(opt.gpu_id)


    lr = opt.lr
    optimizer = Adam([
        {'params': model1.parameters(), 'lr': lr},
        {'params': modelcnn.parameters(), 'lr': lr},
        {'params': modelatt.parameters(), 'lr': lr},
        {'params': capnet.parameters(), 'lr': lr}
    ], lr=opt.lr, betas=(opt.beta1, 0.999))  # 自己



    if opt.resume > 0:
        model1.load_state_dict(torch.load(os.path.join(opt.outf, 'model1_' + str(opt.resume) + '.pt')))
        model1.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf, 'optim1_' + str(opt.resume) + '.pt')))

        modelcnn.load_state_dict(torch.load(os.path.join(opt.outf, 'modelcnn_' + str(opt.resume) + '.pt')))
        modelcnn.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf, 'optimcnn_' + str(opt.resume) + '.pt')))

        modelatt.load_state_dict(torch.load(os.path.join(opt.outf, 'modelatt_' + str(opt.resume) + '.pt')))
        modelatt.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf, 'optimatt_' + str(opt.resume) + '.pt')))


        capnet.load_state_dict(torch.load(os.path.join(opt.outf,'capsule_' + str(opt.resume) + '.pt')))
        capnet.train(mode=True)
        optimizer.load_state_dict(torch.load(os.path.join(opt.outf,'optim_' + str(opt.resume) + '.pt')))

        if opt.gpu_id >= 0:
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(opt.gpu_id)

    if opt.gpu_id >= 0:
        model1.cuda(opt.gpu_id)
        model2.cuda(opt.gpu_id)
        modelcnn.cuda(opt.gpu_id)
        modelatt1.cuda(opt.gpu_id)
        modelatt.cuda(opt.gpu_id)
        capnet.cuda(opt.gpu_id)
        xent.cuda(opt.gpu_id)
        tent.cuda(opt.gpu_id)
        l1loss.cuda(opt.gpu_id)

    transform_fwd = transforms.Compose([
        transforms.Resize((opt.imageSize, opt.imageSize)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
    ])
  

    trainset = MyDatasetff('.../your traindataset', 4,
                          train_transforms)  # 10-frame number


    print(len(trainset))
    train_bs =8

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bs, shuffle=True)

    train_transforms2 = transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.643, 0.466, 0.387), std=(0.259, 0.220, 0.203))
    ])

  

    testset=MyDatasetff('.../your testdataset' ,4,train_transforms2)



    print(len(testset))
    test_bs =8

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bs, shuffle=False)



    for epoch in range(opt.resume+1, opt.niter+1):
        count = 0
        loss_train = 0
        loss_test = 0
        fin_preds = []
        fin_trues = []
        acc_train=-0
        acc_test=0




        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        for batch_idx, data0 in enumerate(trainloader, 1):
            data, target,img_name = data0
            data = Variable(data)
            target = Variable(target)
            target[target > 1] = 1
            img_label = target.numpy().astype(float)
            data, target = data.cuda(opt.gpu_id), target.cuda(opt.gpu_id)
            optimizer.zero_grad()



            img_data = data.cuda(opt.gpu_id)
            labels_data = target.cuda(opt.gpu_id)
            y = torch.repeat_interleave(labels_data , 4)

            img_data = img_data.permute(0, 2, 1, 3, 4)

            b,t,c, w, h = img_data.size()

            img_datas = img_data.contiguous().view(b * t, c, w, h)#torch.Size([64, 3, 224, 224])
            gaze,gaze_bias,feat_map = model1(img_datas)  # (b * t, c, 16, 8)
            gaze2, gaze_bias2, feat_map2 = model2(img_datas)
            
            cnnout,prediction1=modelcnn(img_datas)

            cnnout1 = modelatt(img_datas)
            cnnout1_1=modelatt1(img_datas)


            outputs, features = capnet(img_data,feat_map,cnnout, cnnout1)


            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(xent, outputs, labels_data )

            else:
                xent_loss = xent(outputs, labels_data ).cuda(opt.gpu_id)


            if isinstance(outputs, (tuple, list)):

                tacc = DeepSupervision_acc(accuracy_score, outputs, labels_data)
            else:
                tacc = accuracy_score1(outputs, labels_data).cuda(opt.gpu_id)


            loss1 = l1loss(feat_map, feat_map2)#gaze
            loss2=xent(prediction1,y)#text
            loss3 = l1loss(cnnout1, cnnout1_1)#att
            loss_dis = xent_loss+loss1+loss2+loss3
            loss_dis_data = loss_dis.item()
            loss_dis.backward()
            optimizer.step()
            acc_train += tacc
            loss_train += loss_dis_data
            count += 1

        acc_train/=count
        loss_train/=count


        ########################################################################

        # do checkpointing & validation
        torch.save(model1.state_dict(), os.path.join(opt.outf, 'model1_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim1_%d.pt' % epoch))

        torch.save(modelcnn.state_dict(), os.path.join(opt.outf, 'modelcnn_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optimcnn_%d.pt' % epoch))

        torch.save(modelatt.state_dict(), os.path.join(opt.outf, 'modelatt_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optimatt_%d.pt' % epoch))

        torch.save(capnet.state_dict(), os.path.join(opt.outf, 'capsule_%d.pt' % epoch))
        torch.save(optimizer.state_dict(), os.path.join(opt.outf, 'optim_%d.pt' % epoch))

        model1.eval()
        modelcnn.eval()
        modelatt.eval()
        capnet.eval()

        tol_label = np.array([], dtype=float)
        tol_pred = np.array([], dtype=float)

        count = 0

        for batch_idx, test_data0 in enumerate(testloader, 1):
            test_data, test_target,test_img_name = test_data0
            data = Variable(test_data)
            target = Variable(test_target)
            target[target > 1] = 1
            img_label = target.numpy().astype(float)
            data, target = data.cuda(opt.gpu_id), target.cuda(opt.gpu_id)

            img_data = data.cuda(opt.gpu_id)
            labels_data = target.cuda(opt.gpu_id)
            y = torch.repeat_interleave(labels_data , 4)

            img_data = img_data.permute(0, 2, 1, 3, 4)
            b, t, c, w, h = img_data.size()



            img_datas = img_data.contiguous().view(b * t, c, w, h)

            gaze, gaze_bias ,feat_map = model1(img_datas)  # (b * t, c, 16, 8)
            gaze2, gaze_bias2, feat_map2 = model2(img_datas)



            cnnout,teprediction1=modelcnn(img_datas)
            cnnout1 = modelatt(img_datas)
            cnnout1_1 = modelatt1(img_datas)



            outputs, features = capnet(img_data,feat_map,cnnout,cnnout1 )


            if isinstance(outputs, (tuple, list)):
                xent_loss = DeepSupervision(xent, outputs, labels_data).cuda(opt.gpu_id)

            else:
                xent_loss = xent(outputs, labels_data).cuda(opt.gpu_id)

            if isinstance(outputs, (tuple, list)):

                acc_te = DeepSupervision_acc(accuracy_score, outputs, labels_data)
            else:
                acc_te = accuracy_score1(outputs, labels_data).cuda(opt.gpu_id)

            loss1 = l1loss(feat_map, feat_map2)
            loss2=xent(teprediction1,y)
            loss3 = l1loss(cnnout1, cnnout1_1)
            loss_dis = xent_loss+loss1+loss2+loss3
            loss_dis_data = loss_dis.item()

            acc_test += acc_te
            loss_test+=loss_dis_data
            count += 1

        acc_test /= count
        loss_test /= count





        print('[Epoch %d] Train loss: %.4f   acc: %.2f | Test loss: %.4f  acc: %.2f'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.write('%d,%.4f,%.2f,%.4f,%.2f\n'
        % (epoch, loss_train, acc_train*100, loss_test, acc_test*100))

        text_writer.flush()

        model1.train()
        modelcnn.train()
        modelatt.train()
        capnet.train(mode=True)

    text_writer.close()


