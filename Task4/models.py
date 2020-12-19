import torch
import torch.nn as nn
import config
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,dropout=0):
        super().__init__()
        self.hidden_size=hidden_size
        self.embedding=nn.Embedding(input_size,hidden_size)
        self.gru=nn.GRU(hidden_size,hidden_size,
                        num_layers=num_layers,
                        dropout=0 if num_layers==1 else dropout,
                        bidirectional=True)
        

    def forward(self,input_seq,input_len,hidden=None):
        embedded=self.embedding(input_seq)
        packed_seq=rnn.pack_padded_sequence(embedded,input_len,enforce_sorted=False)
        output_seq,hidden=self.gru(packed_seq,hidden)
        padded_seq,_=rnn.pad_packed_sequence(output_seq)
        output_seq=padded_seq[:,:,:self.hidden_size]+\
                   padded_seq[:,:,self.hidden_size:]
        return output_seq,hidden



class Decoder(nn.Module):
    def __init__(self,output_size,hidden_size,
                 num_layers,method,dropout=0.1):
        super().__init__()
        self.method=method
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.num_layers=num_layers
        self.embedding=nn.Embedding(output_size,hidden_size)
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,
                        num_layers=num_layers,
                        dropout=(0 if num_layers==1 else dropout))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.attention=Attention(method,hidden_size)


    def forward(self,input_step,last_hidden,encoder_output):
        embedding=self.embedding(input_step)
        embedding=self.embedding_dropout(embedding)
        rnn_output,hidden=self.gru(embedding,last_hidden)
        attention_weight=self.attention(rnn_output,encoder_output)
        context=attention_weight.bmm(encoder_output.permute(1,0,2))
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),dim=1)
        concat_output=torch.tanh(self.concat(concat_input))
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)
        return output,hidden



class Attention(nn.Module):
    def __init__(self,method,hidden_size):
        super().__init__()
        self.method=method
        self.hidden_size=hidden_size

        if method not in ['dot','general','concat']:
            raise ValueError(method,'should be dot,general or concat')

        if method=='general':
            self.attention=nn.Linear(hidden_size,hidden_size)
        elif method=='concat':
            self.attention=nn.Linear(hidden_size*2,hidden_size)
            self.v=nn.Parameter(torch.FloatTensor(hidden_size))


    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)


    def general_score(self,hidden,encoder_output):
        energy=self.attention(encoder_output)
        return torch.sum(hidden*energy,dim=2)


    def concat_score(self,hidden,encoder_output):
        concat=torch.cat((hidden.expand(encoder_output.size(0),-1,-1),
                          encoder_output),
                         dim=2)
        energy=torch.tanh(self.attention(concat))
        return torch.sum(self.v*energy,dim=2)


    def forward(self,hidden,encoder_output):
        if self.method=='dot':
            attention_energy=self.dot_score(hidden,encoder_output)
        elif self.method=='general':
            attention_energy=self.general_score(hidden,encoder_output)
        else:
            attention_energy=self.concat_score(hidden,encoder_output)

        attention_energy=attention_energy.permute(1,0)
        return F.softmax(attention_energy,dim=1).unsqueeze(dim=1)



if __name__=='__main__':
    e=Encoder(10761,512,3,dropout=0.3)
