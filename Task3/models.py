import torch
import torch.nn as nn
import torchvision

class UNetConvBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride,padding):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel,
                             kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.relu(self.bn1(self.conv1(x)))
        x=self.relu(self.bn2(self.conv2(x)))
        return x


class UNetBlock(nn.Module):
    def __init__(self,left_in,left_out,
                 sub_layer,
                 right_in,right_out):
        super().__init__()
        self.left_conv=UNetConvBlock(left_in,left_out,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.down=nn.Conv2d(left_out,left_out,
                            kernel_size=2,
                            stride=2,
                            padding=0) 
        self.down=nn.MaxPool2d(kernel_size=2,stride=2)
        self.right_conv=UNetConvBlock(right_in,right_out,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1)
        self.up=nn.Sequential(nn.Conv2d(left_out*2,left_out*4,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0),
                              nn.PixelShuffle(2))
        self.sub_layer=sub_layer

    def forward(self,x):
        left=self.left_conv(x)
        down_left=self.down(left)
        sub=self.sub_layer(down_left)
        right=self.right_conv(torch.cat((left,self.up(sub)),dim=1))
        return right



class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer5=UNetConvBlock(256,512,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)
        self.layer4=UNetBlock(128,256,self.layer5,512,256)
        self.layer3=UNetBlock(64,128,self.layer4,256,128)
        self.layer2=UNetBlock(32,64,self.layer3,128,64)
        self.layer1=UNetBlock(3,32,self.layer2,64,32)
        self.conv=nn.Sequential(nn.Conv2d(32,3,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1),
                                nn.Sigmoid())

    def forward(self,x):
        x=self.conv(self.layer1(x))
        return x


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
