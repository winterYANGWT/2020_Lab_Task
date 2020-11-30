import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image


class Resize(object):
    def __init__(self,size,interpolation=3):
        super().__init__()
        self.size=size
        self.interpolation=interpolation

    def __call__(self,img1,img2):
        img1=F.resize(img1,self.size,self.interpolation)
        img2=F.resize(img2,self.size,self.interpolation)
        return img1,img2


class RandomHorizontalFlip(object):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def __call__(self,img1,img2):
        if torch.rand(1)<self.p:
            return F.hflip(img1),F.hflip(img2)
        
        return img1,img2


class RandomVerticalFlip(object):
    def __init__(self,p=0.5):
        super().__init__()
        self.p=p

    def __call__(self,img1,img2):
        if torch.rand(1)<self.p:
            return F.vflip(img1),F.vflip(img2)
        
        return img1,img2


class ToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self,img1,img2):
        return F.to_tensor(img1),F.to_tensor(img2)


class Compose(object):
    def __init__(self,transforms):
        super().__init__()
        self.transforms=transforms

    def __call__(self,img1,img2):
        for t in self.transforms:
            img1,img2=t(img1,img2)

        return img1,img2


transform_train=Compose([Resize((576,288),3),
                         RandomVerticalFlip(),
                         RandomHorizontalFlip(),
                         ToTensor()])
transform_test=Compose([Resize((576,288),3),
                        ToTensor()])
transform_rgb2tensor=transforms.Compose([transforms.Resize((576,288),3),
                                         transforms.ToTensor()])
transform_tensor2rgb=transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize((600,300),3)])

