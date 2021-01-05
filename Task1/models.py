import torch
import torch.nn as nn



class ResBlock(nn.Module):
    def __init__(self,in_channel,channel,out_channel,stride=1,down_sample=None):
        super().__init__()
        self.down_sample=down_sample
        self.bn1=nn.BatchNorm2d(in_channel)
        self.conv1=nn.Conv2d(in_channel,channel,
                             kernel_size=1,
                             stride=stride,
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
    def __init__(self,input_dim=3,output_dim=2048):
        super().__init__()
        self.conv1=nn.Conv2d(input_dim,64,
                             kernel_size=7,
                             stride=2,
                             padding=3)
        self.pool=nn.AdaptiveAvgPool2d(14)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.conv2=self.make_layers(64,64,256,3)
        self.conv3=self.make_layers(256,128,512,4,stride=2)
        self.conv4=self.make_layers(512,256,1024,6,stride=2)
        self.conv5=self.make_layers(1024,512,output_dim,3)
        

    def make_layers(self,in_channel,channel,output_channel,
                    num_blocks,stride=1):
        layers=[]

        if stride!=1 or in_channel!=output_channel:
            down=nn.Sequential(nn.Conv2d(in_channel,output_channel,
                                         kernel_size=stride,
                                         stride=stride,
                                         padding=0),
                                nn.BatchNorm2d(output_channel))
        
        layers.append(ResBlock(in_channel,channel,output_channel,
                               stride=stride,down_sample=down))
        
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
    def __init__(self,input_dim=3,output_dim=2048):
        super().__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.cnn=ResNet(self.input_dim,self.output_dim)


    def forward(self,x):
        img_feature=self.cnn(x)
        flatten=torch.flatten(img_feature,start_dim=2,end_dim=3)
        flatten=flatten.permute(2,0,1)
        flatten=flatten.contiguous()
        return flatten
    


class Decoder(nn.Module):
    def __init__(self,input_dim,hidden_dim=512,encoder_dim=2048,
                 num_layers=1,dropout=0.1):
        super().__init__()
        self.embedding=nn.Embedding(input_dim,hidden_dim)
        
        self.attention=Attention(encoder_dim,hidden_dim)
        self.lstm=nn.LSTM(input_size=hidden_dim,
                          hidden_size=hidden_dim,
                          num_layers=1)
        self.out=nn.Linear(hidden_dim,input_dim)
        self.h=nn.Linear(encoder_dim+hidden_dim,hidden_dim)
        self.softmax=nn.Softmax(dim=2)
        self.init_c=nn.Linear(encoder_dim,hidden_dim)
        self.init_h=nn.Linear(encoder_dim,hidden_dim)


    def forward(self,input_step,encoder_output,last_h=None,last_c=None):
        if last_h==None:
            last_h=self.init_h(torch.mean(encoder_output,dim=0,keepdim=True))

        if last_c==None:
            last_c=self.init_c(torch.mean(encoder_output,dim=0,keepdim=True))

        embedded=self.embedding(input_step)
        context=self.attention(encoder_output,last_h)
        last_h=self.h(torch.cat([last_h,context],dim=2))
        rnn_output,(h,c)=self.lstm(embedded,(last_h,last_c))
        out=self.softmax(self.out(rnn_output))
        out=out.squeeze(dim=0)
        return out,h,c



class Attention(nn.Module):
    def __init__(self,encoder_hidden_dim,decoder_hidden_dim):
        super().__init__()
        self.v=nn.Parameter(torch.rand(decoder_hidden_dim))
        self.w1=nn.Linear(encoder_hidden_dim,decoder_hidden_dim)
        self.w2=nn.Linear(decoder_hidden_dim,decoder_hidden_dim)
        self.softmax=nn.Softmax(dim=1)


    def forward(self,encoder_output,decoder_hidden):
        decoder_hidden=decoder_hidden.expand((-1,encoder_output.size(1),-1))
        energy=self.w1(encoder_output)+self.w2(decoder_hidden)
        energy=torch.sum(self.v*torch.tanh(energy),dim=2)
        energy=energy.permute(1,0).contiguous()
        weight=self.softmax(energy)
        weight=weight.unsqueeze(1)
        context=weight.bmm(encoder_output.permute(1,0,2).contiguous())
        context=context.permute(1,0,2).contiguous()
        return context



if __name__=='__main__':
    e=Encoder()
    d=Decoder(10000)

