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
    train_data=datasets.Flickr8k
    train_data_loader=data.DataLoader(dataset=train_data,
                                      batch_size=config.BATCH_SIZE,
                                      shuffle=True,
                                      collate_fn=datasets.collate_fn)

    #load model
    encoder=models.Encoder()
    decoder=models.Decoder(train_data.num_word)

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
                             lr=config.LR)

    #set loss and meter
    criterion=losses.MaskLoss()
    loss_meter=metrics.LossMeter()
    bleu_meter=metrics.BLEUMeter(max_n=4)
    
    #train
    encoder.train()
    decoder.train()

    for epoch in range(config.EPOCH_START,config.EPOCH):
        with tqdm(total=len(train_data),ncols=100) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1,config.EPOCH))

            loss_meter.reset()
            bleu_meter.reset()

            for image,caption,max_len,mask in train_data_loader:
                num_batch=image.size(0)
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                loss=0

                encoder_output=encoder(image)

                decoder_input=torch.LongTensor([[1 for _ in range(num_batch)]])
                decoder_input=decoder_input.to(config.DEVICE)
                h=None
                c=None
                total_output=[]

                use_teacher_forcing=random.random()<config.TEACHER_FORCING_RATIO

                for step in range(max_len):
                    decoder_output,h,c=decoder(decoder_input,
                                               encoder_output,
                                               h,c)

                    _,indices=torch.topk(decoder_output,k=1)
                    total_output.append(indices)

                    if use_teacher_forcing==True:
                        decoder_input=caption[step].view(1,-1).contiguous()
                    else:
                        decoder_input=indices.permute(1,0).contiguous()

                    decoder_input=decoder_input.to(config.DEVICE)

                    current_loss=criterion(decoder_output,caption[step],mask[step])
                    loss+=current_loss
                    count=torch.sum(mask[step])
                    loss_meter.update(current_loss.item(),count.item())

                index_list=torch.cat(total_output,dim=1).unsqueeze(1)
                target_list=caption.permute(1,0).unsqueeze(1)
                index_list=utils.tensor2list(index_list)
                target_list=utils.tensor2list(target_list)
                bleu_meter.update(index_list,target_list)
                loss.backward()
                nn.utils.clip_grad_norm_(encoder.parameters(),config.CLIP)
                nn.utils.clip_grad_norm_(decoder.parameters(),config.CLIP)
                encoder_optim.step()
                decoder_optim.step()
                t.set_postfix(loss='{:.6f}'.format(loss_meter.value),
                              bleu='{:.6f}'.format(bleu_meter.value))
                t.update(num_batch)

                utils.save_model(encoder,
                                 os.path.join('./Model',str(epoch+1)),
                                 'encoder.pth')
                utils.save_model(decoder,
                                 os.path.join('./Model',str(epoch+1)),
                                 'decoder.pth')

