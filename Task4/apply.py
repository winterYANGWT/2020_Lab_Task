import torch
import torch.nn as nn
import models
import datasets
import utils



class GreedySearchBot(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder


    def forward(self,input_seq,input_len,max_len):
        encoder_output,encoder_hidden=self.encoder(input_seq,input_len)
        decoder_hidden=encoder_hidden[:self.encoder.num_layers]
        decoder_input=torch.LongTensor([[1]])
        output=torch.LongTensor([0])
        output_score=torch.Tensor([0])

        for i in range(max_len):
            decoder_output,decoder_hidden=self.decoder(decoder_input,
                                                       decoder_hidden,
                                                       encoder_output)
            decoder_score,decoder_input=torch.max(decoder_output,dim=1)
            output_score=torch.cat((output_score,decoder_score),dim=0)
            output=torch.cat((output,decoder_input).dim=0)
            decoder_input=decoder_input.unsqueeze(0)

        return output,output_score



def process(bot,input_sentence,word2index,index2word,max_len):
    sentence_len=len(input_sentence)

    if sentence_len>max_len-1:
        input_sentence=input_sentence[:max_len-1]
        sentence_len=max_len-1

    input_sentence.append(2)
    sentence_len+=1
    sentence_int=[]

    for word in input_sentence:
        sentence2list.append(voc_dict[word])

    input_seq=torch.LongTensor([sentence2list]).permute(1,0)
    input_len=torch.LongTensor([sentence_len])

    output,output_score=bot(input_seq,input_len,max_len)
    sentence_str=''

    for index for output:
        word=index2word[index]

        if word=='EOS':
            break
        elif word!='PAD':
            sentence_str+=word

    return sentence_str
        

if __name__=='__main__':
    dataset=datasets.Cornell
    encoder=models.encoder(dataset.num_word,512,2,0.1)
    decoder=models.decoder(dataset.num_word,512,2,'dot',0.1)
    utils.load_model(encoder,os.path.join('./Model',str(config.MODEL)),'encoder.pth')
    utils.load_model(decoder,os.path.join('./Model',str(config.MODEL)),'decoder.pth')

    bot=GreedySearchBot(encoder,decoder)
    index2word=dataset.index2word
    word2index=dataset.word2index
    max_len=10

    while(True):
        input_sentence=input('>>> ')

        if input_sentence=='q':
            break
        else:
            result=process(bot,input_sentence,word2index,index2word,max_len)
            print(result)

