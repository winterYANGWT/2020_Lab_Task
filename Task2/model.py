import config
import torch
import torch.nn as nn
import torchvision
import math


class BackboneBasicBlock(nn.Module):
    def __init__(self,input_size,output_size,kernel_size,stride,padding):
        super().__init__()
        self.stride=stride
        self.conv1=nn.Conv2d(input_size,output_size,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn1=nn.BatchNorm2d(output_size)
        self.conv2=nn.Conv2d(output_size,output_size,
                             kernel_size=kernel_size,
                             stride=1,
                             padding=padding)
        self.bn2=nn.BatchNorm2d(output_size)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=nn.Sequential(nn.Conv2d(input_size,output_size,
                                                kernel_size=1,
                                                stride=stride,
                                                padding=0,
                                                bias=False),
                                      nn.BatchNorm2d(output_size))

    def forward(self,x):
        _x=x
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.bn2(self.conv2(x))
        
        if self.stride!=1:
            _x=self.downsample(_x)

        x=self.relu(x+_x)
        return x


class BackboneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,
                             kernel_size=7,
                             stride=2,
                             padding=3)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,
                                  stride=2,
                                  padding=1)
        self.conv2=self.make_layers(64,64,3)
        self.conv3=self.make_layers(64,128,4)
        self.conv4=self.make_layers(128,256,6)
        self.conv5=self.make_layers(256,512,3)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x
        
    def make_layers(self,input_size,output_size,layer_num):
        layers=[]
        layers.append(BackboneBasicBlock(input_size,output_size,
                                         kernel_size=3,
                                         stride=2,
                                         padding=1))
        
        for i in range(layer_num-1):
            layers.append(BackboneBasicBlock(output_size,output_size,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1))


        return nn.Sequential(*layers)
        

class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ratio=[0.5,1,2]
        self.anchor_scales=[8,16,32]
        self.feature_stride=16
        self.conv1=nn.Conv2d(512,512,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.cls_conv=nn.Conv2d(512,3*3*2,
                                kernel_size=1,
                                stride=1,
                                padding=0)
        self.reg_conv=nn.Conv2d(512,3*3*4,
                                kernel_size=1,
                                stride=1,
                                padding=0)

    def forward(self,features):
        features=self.feature_extration(x)
        small_features=self.conv1(features)
        
        cls=self.cls_conv(small_features)
        reg=self.reg_conv(small_features)

    def generate_anchor(self,base_size,ratios,anchor_scales):
        anchors=torch.zeros([len(ratios)*len(anchor_scales),4])
        y=base_size/2
        x=base_size/2
        i=0

        for ratio in ratios:
            for anchor_scale in anchor_scales:
                h=base_size*anchor_scale*math.sqrt(ratio)
                w=base_size*anchor_scale*math.sqrt(1/ratio)
                anchors[i]=torch.tensor([x-w/2,y-h/2,x+w/2,y+h/2])

        return anchors


class FasterRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extraction=BackboneNet()

    def forward(self,x):
        x=self.feature_extraction(x)
        return x


if __name__ == '__main__':
    model=FasterRCNN()
