import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

from models.backbone.resnet import *
from models.STAM import STAM,STAM1,STAM2

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def init_pretrained_weight(model, model_url):
    """Initializes model with pretrained weight

    Layers that don't match with pretrained layers in name or size are kept unchanged
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weight_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class PSTA(nn.Module):

    def __init__(self, num_classes, seq_len=12):
        super(PSTA, self).__init__()

        self.in_planes = 2048
        self.base = ResNet()


        self.seq_len = seq_len
        self.num_classes = num_classes
        self.plances = 1024
        self.mid_channel = 256

        self.avg_2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.down_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_planes, out_channels=self.plances, kernel_size=1, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(self.plances),
            self.relu
        )

        t = seq_len
        # print(t,'tttt')
        self.layer1 = STAM2(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 3, num='1')

        # self.layer1 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=4, num='1')#ziji

        t = t / 2
        self.layer2 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 2, num='2')

        t = t / 2
        # print(t,'t')
        self.layer3 = STAM(inplanes=self.plances, mid_planes=self.mid_channel, seq_len=t / 3, num='3')

        self.bottleneck = nn.ModuleList([nn.BatchNorm1d(self.plances) for _ in range(3)])
        self.classifier = nn.ModuleList([nn.Linear(self.plances, num_classes) for _ in range(3)])

        self.bottleneck[0].bias.requires_grad_(False)
        self.bottleneck[1].bias.requires_grad_(False)
        self.bottleneck[2].bias.requires_grad_(False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weight_init_classifier)



    def forward(self, x, feat_map,cnnout,cnnout1,pids=None, camid=None):
        b, t, c, w1, h2 = x.size()
        w = feat_map.size(2)
        # print(w,'w')
        h = feat_map.size(3)

        feat_map = self.down_channel(feat_map)
        feat_map = feat_map.view(b, t, -1, w, h)
        #print(feat_map.shape,"feat_map.shape")#torch.Size([8, 8, 1024, 7, 7]) feat_map.shape


        cnnout = self.down_channel(cnnout)
        cnnout = cnnout.view(b, t, -1, w, h)

        cnnout1 = self.down_channel(cnnout1)
        cnnout1 = cnnout1.view(b, t, -1, w, h)

        fe_cat=[]

        nn=[]
        bs0, t,c,  w, h = feat_map.shape[0:5]

        for ii in range(t):
            feat_x = feat_map[:,  ii,:, :, :]  # [64, 3, 224, 224])
            #print(feat_x.shape)#torch.Size([8, 1024, 7, 7])
            fe_cat.append(feat_x)
            nn.append(feat_x)
        cnn_cat = []
        bs0, t, c, w, h = cnnout.shape[0:5]

        for ii in range(t):
            cnn_x = cnnout[:, ii, :, :, :]  # [64, 3, 224, 224])
            #print(cnn_x.shape)  # torch.Size([8, 1024, 7, 7])
            cnn_cat.append(cnn_x)
            nn.append(cnn_x)
        cnn_cat1 = []
        bs0, t, c, w, h = cnnout1.shape[0:5]

        for ii in range(t):
            cnn_x1 = cnnout1[:, ii, :, :, :]  # [64, 3, 224, 224])
            # print(cnn_x.shape)  # torch.Size([8, 1024, 7, 7])
            cnn_cat1.append(cnn_x1)
            nn.append(cnn_x1)



        new_list = [elem for pair in zip(fe_cat,  cnn_cat,cnn_cat1) for elem in pair]
        #print(len(new_list),'newlist')
        # new_list=nn

        # print(new_list,'new')

        # 在第 1 维进行拼接
        new_feat = torch.stack(new_list, dim=1)
        #print(new_feat.shape,'new_feat.shape')#torch.Size([8, 16, 1024, 7, 7]) new_feat.shape








        feature_list = []
        list = []
        # print(new_feat.shape,'new_feat.shape')

        # print(feat_map.shape,'2')#torch.Size([4, 8, 1024, 7, 7]) 2
        feat_map_1 = self.layer1(new_feat)
        #print(feat_map_1.shape,'feat_map_1.shape')#torch.Size([8, 4, 1024, 7, 7]) feat_map_1.shape

        feature_1 = torch.mean(feat_map_1, 1)
        feature1 = self.avg_2d(feature_1).view(b, -1)
        feature_list.append(feature1)
        # print(feature1.shape,feat_map_1.shape,'f1')
        list.append(feature1)

        feat_map_2 = self.layer2(feat_map_1)
        feature_2 = torch.mean(feat_map_2, 1)
        feature_2 = self.avg_2d(feature_2).view(b, -1)
        list.append(feature_2)

        feature2 = torch.stack(list, 1)
        feature2 = torch.mean(feature2, 1)
        feature_list.append(feature2)
        # print(feature2.shape,feat_map_2.shape, 'f2.shape')

        feat_map_3 = self.layer3(feat_map_2)
        feature_3 = torch.mean(feat_map_3, 1)
        feature_3 = self.avg_2d(feature_3).view(b, -1)
        list.append(feature_3)

        feature3 = torch.stack(list, 1)
        feature3 = torch.mean(feature3, 1)
        # print(feature3.shape, feat_map_3.shape, 'f3.shape')
        feature_list.append(feature3)

        BN_feature_list = []
        for i in range(len(feature_list)):
            BN_feature_list.append(self.bottleneck[i](feature_list[i]))#torch.Size([8, 1024]),torch.Size([8, 2])
        torch.cuda.empty_cache()

        cls_score = []
        confidence = []
        labels=[]
        fina=[]

        # real和fake分数距离绝对值作为权重，求加权平均分数
        def score_fusion(real_score, fake_score, distance):
            weight = 1 / (distance + 1e-8)  # 避免分母为0
            weighted_score = (real_score * weight + fake_score * weight) / 2
            return weighted_score

        # 判断正例还是负例
        def predict_label(score, threshold=None):
            if threshold is None:
                threshold = torch.median(score)
            result = torch.zeros_like(score)
            result[score >= threshold] = 1
            return result
        for i in range(len(BN_feature_list)):
            cls_score.append(self.classifier[i](BN_feature_list[i]))
            # print(BN_feature_list[i].shape,BN_feature_list[i],self.classifier[i](BN_feature_list[i]).shape,self.classifier[i](BN_feature_list[i]),'gdsfd')
        for i in range(len(cls_score)):
            t=torch.softmax(cls_score[i], dim=1)
            confidences, predicted_labels = torch.max(t, 1)  # 找到最大概率对应的类别标签和该类别的概率值
            # print(t.shape,t,'tttt')#torch.Size([8, 2])
            # print(confidences.shape,confidences,'con')
            # print(predicted_labels.shape,predicted_labels,'pre')
            confidence.append(torch.softmax(cls_score[i], dim=1))
            #print(confidence)

        return cls_score,confidence

       # return cls_score, BN_feature_list


