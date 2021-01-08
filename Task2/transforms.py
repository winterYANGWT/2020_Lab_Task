import torch
import torchvision.transforms as transforms
from PIL import Image


class BboxToTensor(object):
    def __init__(self):
        super().__init__()


    def __call__(self,l):
        tensor=torch.LongTensor(l)
        return tensor



class LabelToTensor(object):
    def __init__(self):
        super().__init__()


    def __call__(self,l):
        tensor=torch.LongTensor(l)
        return tensor


        
transform_img_train=transforms.Compose([transforms.Resize([800,600],Image.BICUBIC),
                                        transforms.ColorJitter(brightness=0.1,
                                                               contrast=0.1,
                                                               saturation=0.1,
                                                               hue=0.1),
                                        transforms.ToTensor()])
transform_test=transforms.Compose([transforms.Resize([800,600],Image.BICUBIC),
                                   transforms.ToTensor()])
transform_bbox=BboxToTensor()
transform_label=LabelToTensor()
