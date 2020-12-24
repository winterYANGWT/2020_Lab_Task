import torch
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self,in_channel,channel,out_channel,down_sample=None):
        super().__init__()
        self.down_sample=down_sample
        self.bn1=nn.BatchNorm(in_channel)
        self.conv1=nn.Conv2d(in_channel,channel,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.bn2=nn.BatchNorm2d(channel)
        self.conv2=nn.Conv2d(channel,channel,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        self.bn3=nn.BatchNorm2d(channel)
        self.conv3=nn.Conv2d(channel,out_channel,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.activation=nn.ReLU(inplace=True)


    def forward(self,x):
        identity=x

        if self.down_sample!=None:
            identity=self.down_sample(identity)

        x=self.bn1(x)
        x=self.activation(x)
        x=self.conv1(x)
        x=self.bn2(x)
        x=self.activation(x)
        x=self.conv2(x)
        x=self.bn3(x)
        x=self.activation(x)
        x=self.conv3(x)
        return x+identity



class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,
                             kernel_size=7,
                             stride=2,
                             padding=3)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2)
        self.conv2=self.make_layers(64,64,256,3)
        self.conv3=self.make_layers(256,128,512,4,stride=2)
        self.conv4=self.make_layers(512,256,1024,6,stride=2)
        self.conv5=self.make_layers(1024,512,2048,3)
        

    def make_layers(self,in_channel,channel,output_channel,
                    num_blocks,stride=1):
        layers=[]

        if stride!=1 or in_channel!=output_channel:
            down=nn.Sequential(nn.Conv2d(in_channel,output_channel,
                                         kernel_size=2,
                                         stride=stride,
                                         padding=0),
                                nn.BatchNorm2d(output_channel))
        
        layers.append(ResBlock(in_channel,channel,output_channel,
                               down_sample=down))
        
        for _ in range(1,num_blocks):
            layers.append(ResBlock(output_channel,channel,output_channel))

        return nn.Sequential(*layers)

    
    def forward(self,x):
        x=self.maxpool(self.conv1(x))
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        return x



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn=ResNet()


    def forward(x):
        img_feature=self.cnn(x)
        img_feature=img_feature.permute(0,2,3,1)
        flatten=torch.flatten(img_feature,start_dim=1,end_dim=2)
        return img_feature
    


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()



class Attention(nn.Module):
    def __init__(self,encoder_hidden_size,decoder_hidden_size):
        super().__init__()
        self.v=nn.Parameter(torch.rand(decoder_hidden_size))
        self.w1=nn.Linear(encoder_hidden_size,decoder_hidden_size)
        self.w2=nn.Linear(decoder_hidden_size,decoder_hidden_size)
        self.softmax=nn.Softmax(dim=1)


    def forward(self,encoder_output,decoder_hidden):
        decoder_hidden=decoder_hidden.expand((-1,encoder_output.size(1),-1))
        energy=self.w1(encoder_hidden)+self.w2(decoder_hidden)
        energy=self.v*torch.tanh(energy)
        weight=self.softmax(energy)
        weight=weight.unsqueeze(1)
        context=weight.bmm(encoder_output)
        context=context.squeeze(1)
        return context

