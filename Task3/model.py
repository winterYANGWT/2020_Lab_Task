import torch
import torch.nn as nn
import torchvision

class UNetConvBlock(nn.Module):
    def __init__(self,in_channel,mid_channel,out_channel,kernel_size,stride,padding):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,mid_channel,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn1=nn.BatchNorm2d(mid_channel)
        self.conv2=nn.Conv2d(mid_channel,out_channel,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=UNetConvBlock(3,64,64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.downsample1=nn.Conv2d(64,64,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
        self.conv2=UNetConvBlock(64,128,128,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.downsample2=nn.Conv2d(128,128,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)
#       self.conv3=UNetConvBlock(128,256,256,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)
#        self.conv4=UNetConvBlock(256,512,512,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)
        self.conv5=UNetConvBlock(128,256,256,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
#        self.conv6=UNetConvBlock(1024,512,256,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)
#        self.conv7=UNetConvBlock(512,256,256,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1)
        self.upsample2=nn.Sequential(nn.Conv2d(256,128*4,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.PixelShuffle(2))
        self.conv8=UNetConvBlock(256,128,128,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.upsample1=nn.Sequential(nn.Conv2d(128,64*4,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     nn.PixelShuffle(2))
        self.conv9=UNetConvBlock(128,64,64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.conv10=nn.Conv2d(64,3,
                              kernel_size=1,
                              stride=1)

    def forward(self,x):
        left1=self.conv1(x)
        down_left1=self.downsample1(left1)
        left2=self.conv2(down_left1)
        down_left2=self.downsample2(left2)
#        left3=self.conv3(down_left2)
#        left4=self.conv4(left3)
        mid=self.conv5(down_left2)
#        right4=self.conv6(torch.cat((left4,mid),dim=1))
#        right3=self.conv7(torch.cat((left3,mid),dim=1))
        right2=self.conv8(torch.cat((left2,self.upsample2(mid)),dim=1))
        right1=self.conv9(torch.cat((left1,self.upsample1(right2)),dim=1))
        result=self.conv10(right1)
        return result


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(6,64,
                             kernel_size=4,
                             stride=2,
                             padding=1)
        self.conv2=nn.Conv2d(64,128,
                             kernel_size=4,
                             stride=2,
                             padding=1)
        self.bn2=nn.BatchNorm2d(128)
        self.conv3=nn.Conv2d(128,256,
                             kernel_size=4,
                             stride=2,
                             padding=1)
        self.bn3=nn.BatchNorm2d(256)
        self.conv4=nn.Conv2d(256,512,
                             kernel_size=4,
                             stride=2,
                             padding=1)
        self.bn4=nn.BatchNorm2d(512)
        self.conv5=nn.Conv2d(512,1,
                             kernel_size=4,
                             stride=1,
                             padding=1)
        self.leaky_relu=nn.LeakyReLU(0.2,inplace=True)
        self.sigmoid=nn.Sigmoid()

    def forward(self,img1,img2):
        x=torch.cat((img1,img2),dim=1)
        x=self.leaky_relu(self.conv1(x))
        x=self.leaky_relu(self.bn2(self.conv2(x)))
        x=self.leaky_relu(self.bn3(self.conv3(x)))
        x=self.leaky_relu(self.bn4(self.conv4(x)))
        x=self.sigmoid(self.conv5(x))
        return x


if __name__=='__main__':
    m=UNet()
    n=PatchDiscriminator()
