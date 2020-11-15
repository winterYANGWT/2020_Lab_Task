import torch
import torch.nn as nn

class PSNR(object):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,img1,img2):
        PSNR=self.calc_PSNR(img1,img2)
        self.count+=img1.size()[0]
        self.sum+=PSNR*img1.size()[0]
        self.avg=self.sum/self.count

    def get_avg(self):
        return self.avg

    def calc_PSNR(self,img1,img2):
        with torch.no_grad():
            mse=nn.functional.mse_loss(img1,img2)
            psnr=10*torch.log(1/mse)

        return psnr.item()
