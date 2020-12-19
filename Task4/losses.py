import torch
import torch.nn as nn



class MaskLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.cel=nn.NLLLoss(reduction='none')


    def forward(self,output,target,mask):
        output=torch.log(output)
        cross_entropy_loss=self.cel(output,target)
        return cross_entropy_loss.masked_select(mask).mean()

