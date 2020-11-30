import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import config
import os
import models
import datasets
import losses
import metrics


if __name__=='__main__':
    cudnn.benchmark=True
    #load data
    train_data=datasets.maps_train
    val_data=datasets.maps_test
    train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                  batch_size=config.BATCH_SIZE,
                                                  shuffle=True)
    val_data_loader=torch.utils.data.DataLoader(dataset=val_data,
                                                batch_size=config.BATCH_SIZE)

    #load model
    G=models.UNet().to(config.DEVICE)
    D=models.PatchDiscriminator().to(config.DEVICE)
    
    #set optimizer
    G_optim=optim.Adam(G.parameters(),
                       lr=config.LEARNING_RATE,
                       betas=config.BETAS)
    D_optim=optim.Adam(D.parameters(),
                       lr=config.LEARNING_RATE,
                       betas=config.BETAS)
 
    #set criterion
    G_criterion=losses.GLoss()
    D_criterion=losses.DLoss()
    
    #set meter
    PSNR_meter=metrics.PSNR()

    #train
    G.train()
    D.train()
    for epoch in range(config.EPOCHES):
        with tqdm(total=len(train_data),ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch+1,config.EPOCHES))

            for input_img,real_img in train_data_loader:
                fake_img=G(input_img)
                real_pred=D(input_img,real_img)
                fake_pred=D(input_img,fake_img)

                G_loss=G_criterion(fake_pred,real_img,fake_img)
                G_optim.zero_grad()
                G_loss.backward()
                G_optim.step()

                fake_pred=D(input_img,fake_img.detach())
                D_loss=D_criterion(real_pred,fake_pred)
                D_optim.zero_grad()
                D_loss.backward()
                D_optim.step()
    
                PSNR_meter.update(input_img,fake_img)
                
                t.set_postfix(PSNR='{:.6f}'.format(PSNR_meter.get_avg()))
                t.update(len(input_img))

        save_model(G,os.path.join('./Model',str(epoch+1),'G.mdl'))
        save_model(D,os.path.join('./Model',str(epoch+1),'D.mdl'))

