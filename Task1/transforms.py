import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image



def Pad(object):
    def __init__(self):
        super().__init__(fill_size)
        self.fill_size


    def __call__(self,img):
        w,h=img.size()
        left=(self.fill_size[0]-w)//2
        top=(self.fill_size[1]-h)//2
        right=self.fill_size[0]-w-left
        bottom=self.fill_size[1]-h-top
        return F.pad(img,(left,top,right,bottom))



image_transform=transforms.Compose()
