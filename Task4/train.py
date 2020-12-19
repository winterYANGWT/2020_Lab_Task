import torch
import torch.nn as nn
import torch.optim as optim
import random
from tqdm import tqdm
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import os
import config
import models
import datasets
import losses
import metrics
import utils



if __name__=='__main__':
    cudnn.benchmark=True

    #load data
    train_data=datasets.Cornell
    train_data_loader=data.DataLoader(dataset=train_data,
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=True,
                                      collate_fn=datasets.collate_fn)

    #load model
    encoder=models.Encoder(train_data.num_word,512,2,dropout=0.1)
    decoder=models.Decoder(train_data.num_word,512,2,'dot',dropout=0.1)
    if config.LOAD==True:
        utils.load_model(encoder,
                         os.path.join('./Model',str(config.EPOCH_START)),
                         'encoder.pth')
        utils.load_model(decoder,
                         os.path.join('./Model',str(config.EPOCH_START)),
                         'decoder.pth')

    #set optimizer
    encoder_optim=optim.Adam(encoder.parameters(),
                             lr=config.LR)
    decoder_optim=optim.Adam(decoder.parameters(),
                             lr=config.LR*5)

    #set loss and meter
    criterion=losses.MaskLoss()
    loss_meter=metrics.LossMeter()

    #train
    encoder.train()
    decoder.train()

    for epoch in range(config.EPOCH_START,config.EPOCH):
        with tqdm(total=len(train_data),ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1,config.EPOCH))

            loss_meter.reset()

            for input_seq,input_len,output_seq,output_len,mask in train_data_loader:
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                loss=0
                
                encoder_output,encoder_hidden=encoder(input_seq,input_len)

                decoder_input=torch.LongTensor([[1 for _ in range(input_seq.size(1))]])
                decoder_input=decoder_input.to(config.DEVICE)
                decoder_hidden=encoder_hidden[:decoder.num_layers]

                use_teacher_forcing=random.random()<config.TEACHER_FORCING_RATIO

                for step in range(output_len):
                    decoder_output,decoder_hidden=decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_output)
                    
                    if use_teacher_forcing==True:
                        decoder_input=output_seq[step].view(1,-1)
                    else:
                        _,indices=torch.topk(decoder_output,k=1)
                        decoder_input=indices.permute(1,0).to(config.DEVICE)

                    current_loss=criterion(decoder_output,output_seq[step],mask[step])
                    loss+=current_loss
                    count=torch.sum(mask[step])
                    loss_meter.update(current_loss.item(),count.item())

                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(),config.CLIP)
                nn.utils.clip_grad_norm_(decoder.parameters(),config.CLIP)
                encoder_optim.step()
                decoder_optim.step()
                t.set_postfix(loss='{:.6f}'.format(loss_meter.value))
                t.update(input_seq.size()[1])

            utils.save_model(encoder,os.path.join('./Model',str(epoch+1)),'encoder.pth')
            utils.save_model(decoder,os.path.join('./Model',str(epoch+1)),'decoder.pth')
