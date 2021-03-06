import torch
import torch.nn as nn
import config


class DLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCEWithLogitsLoss().to(config.DEVICE)

    def forward(self,real_pred,fake_pred):
        real_loss=self.bce(real_pred,torch.ones(real_pred.size()).to(config.DEVICE))
        fake_loss=self.bce(fake_pred,torch.zeros(fake_pred.size()).to(config.DEVICE))
        print('D')
        print(real_loss,fake_loss)
        return real_loss+fake_loss


class GLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()
        self.bce=nn.BCEWithLogitsLoss().to(config.DEVICE)
        self.smooth_l1=nn.MSELoss().to(config.DEVICE)

    def forward(self,fake_pred,real_img,fake_img):
        adversarial_loss=self.bce(fake_pred,torch.ones(fake_pred.size()).to(config.DEVICE))
        content_loss=self.smooth_l1(real_img,fake_img)
        print('G')
        print(adversarial_loss,content_loss)
        return adversarial_loss+config.L1_LAMBDA*content_loss
      #  return config.L1_LAMBDA*content_loss                                     
