import torch
import torch.nn as nn



class MaskLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        cel=nn.NLLLoss(reduce=False,reduction='none')


    def forward(output,target,mask):
        cross_entropy_loss=cel(output,target)
        return cross_entropy_loss.masked_select(mask).mean()

