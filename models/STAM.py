import torch
import torch.nn as nn
from models.SRA import SRA,SRA1,SRA2
from models.TRA import TRA,TRA1,TRA2

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

class STAM(nn.Module):

    def __init__(self, inplanes, mid_planes, num, **kwargs):

        super(STAM, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.num = num

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        self.TRAG = TRA(inplanes=inplanes, num=num)
        self.SRAG = SRA(inplanes=inplanes, num=num)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        #print(feat_map.size(),'feat_map.size()')#torch.Size([4, 8, 1024, 7, 7]) feat_map.size()

        reshape_map = feat_map.view(b * t, c, h, w)
        feat_vect = self.avg(reshape_map).view(b, t, -1)
        #print(feat_vect.shape,'feat_vect')#torch.Size([4, 8, 1024]) feat_vect

        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)
        #print(embed_feat.shape,'embed_feat.shape')

        gap_feat_map0 = self.TRAG(feat_map, reshape_map, feat_vect, embed_feat)
        gap_feat_map = self.SRAG(feat_map, reshape_map, embed_feat, feat_vect, gap_feat_map0)
        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, -1, c, h, w)
        torch.cuda.empty_cache()

        return gap_feat_map



class STAM1(nn.Module):

    def __init__(self, inplanes, mid_planes, num, **kwargs):

        super(STAM1, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.num = num

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        self.TRAG = TRA1(inplanes=inplanes, num=num)
        self.SRAG = SRA1(inplanes=inplanes, num=num)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        #print(feat_map.size(),'feat_map.size()')#torch.Size([4, 8, 1024, 7, 7]) feat_map.size()

        reshape_map = feat_map.view(b * t, c, h, w)
        feat_vect = self.avg(reshape_map).view(b, t, -1)
        #print(feat_vect.shape,'feat_vect')#torch.Size([4, 8, 1024]) feat_vect

        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)
        print(embed_feat.shape,'embed_feat.shape')

        gap_feat_map0 = self.TRAG(feat_map, reshape_map, feat_vect, embed_feat)
        print(gap_feat_map0.shape,'gap_feat_map0.shape')#torch.Size([8, 4, 1024, 7, 7]) gap_feat_map0.shape

        gap_feat_map = self.SRAG(feat_map, reshape_map, embed_feat, feat_vect, gap_feat_map0)
        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, -1, c, h, w)
        torch.cuda.empty_cache()

        return gap_feat_map


class STAM2(nn.Module):

    def __init__(self, inplanes, mid_planes, num, **kwargs):

        super(STAM2, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.num = num

        self.Embeding = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=128,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(128),
            self.relu
        )
        self.Embeding.apply(weights_init_kaiming)

        self.TRAG = TRA2(inplanes=inplanes, num=num)
        self.SRAG = SRA2(inplanes=inplanes, num=num)

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=inplanes, out_channels=mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=mid_planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_planes),
            self.relu,
            nn.Conv2d(in_channels=mid_planes, out_channels=inplanes, kernel_size=1, bias=False),
            nn.BatchNorm2d(inplanes),
            self.relu
        )
        self.conv_block.apply(weights_init_kaiming)

    def forward(self, feat_map):

        b, t, c, h, w = feat_map.size()
        #print(feat_map.size(),'feat_map.size()')#torch.Size([4, 8, 1024, 7, 7]) feat_map.size()

        reshape_map = feat_map.view(b * t, c, h, w)
        feat_vect = self.avg(reshape_map).view(b, t, -1)
        #print(feat_vect.shape,'feat_vect')#torch.Size([4, 8, 1024]) feat_vect

        embed_feat = self.Embeding(reshape_map).view(b, t, -1, h, w)
        # print(embed_feat.shape,'embed_feat.shape')#torch.Size([8, 12, 128, 7, 7])

        gap_feat_map0 = self.TRAG(feat_map, reshape_map, feat_vect, embed_feat)
        # print(gap_feat_map0.shape,'gap_feat_map0.shape')#torch.Size([8, 1, 1024, 7, 7])
        gap_feat_map = self.SRAG(feat_map, reshape_map, embed_feat, feat_vect, gap_feat_map0)
        gap_feat_map = self.conv_block(gap_feat_map)
        gap_feat_map = gap_feat_map.view(b, -1, c, h, w)
        torch.cuda.empty_cache()

        return gap_feat_map