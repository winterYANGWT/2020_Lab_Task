import config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import model
import datasets
import losses
import metrics


if __name__=='__main__':
    #initial
    cudnn.benchmark=True
    train_data=datasets.VOC12_train
    val_data=datasets.VOC12_val

    #load model
    FasterRCNN=model.FasterRCNN()

    #running
    for epoch in range(config.EPOCH_START,config.EPOCH_START+config.EPOCH):
        #load data
        train_data_loader=torch.utils.data.DataLoader(dataset=train_data,
                                                      batch_size=config.BATCH_SIZE,
                                                      shuffle=True,
                                                      collate_fn=datasets.collate_fn,
                                                      num_workers=4)
        val_data_loader=torch.utils.data.DataLoader(dataset=val_data,
                                                    batch_size=config.BATCH_SIZE,
                                                    shuffle=False,
                                                    num_workers=4)

        for images,targets in train_data_loader:
            print(images.shape)
            print(targets)

        for images,targets in val_data_loader:
            a=[]
