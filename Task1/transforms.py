import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image



class Pad(object):
    def __init__(self):
        super().__init__(fill_size)
        self.fill_size


    def __call__(self,img):
        w,h=img.size()
        width=(self.fill_size[0]-w)//2
        height=(self.fill_size[1]-h)//2
        return F.pad(img,(width,height))



class TextToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self,caption,caption_len,caption_mask):
        caption=torch.LongTensor(caption)
        caption_len=torch.LongTensor(caption_len)
        caption_mask=torch.BoolTensor(caption_mask)
        return caption,caption_len,caption_mask



image_transform=transforms.Compose([Pad((500,500))],
                                    transforms.Resize(224,224)
                                    transforms.ToTensor())
text_transfrom=TextToTensor()
   
