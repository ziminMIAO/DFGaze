import torch.nn as nn
import torchvision
import torch


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.base = torchvision.models.resnet34(pretrained=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 添加一个全局平均池化层
        self.fc = nn.Linear(512,2048)  # 添加一个全连接层，将 512 维特征向量映射为 2048 维特征向量

        # 加载预训练权重
        state_dict = torch.load('/data/Capsule-Forensics-v2-master (1)/attribute model/res34_fair_align_multi_4_20190809.pt' )
        self.base.load_state_dict(state_dict)
        # 修改分类层，输出18维的特征向量
        self.base.fc = nn.Linear(self.base.fc.in_features, 18)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x11 = self.base.layer4(x)
        b,c,h,w=x11.size()
        x1=x11.view(b*h*w,c)
        x1 = self.fc(x1)
        x1=x1.view(b,-1,h,w)


        x = self.avgpool(x11)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x1
