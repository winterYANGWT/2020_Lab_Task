import torch
import torchvision.transforms as transforms
from PIL import Image


class TargetToTensor(object):
    def __init__(self):
        super().__init__()

    def parser(self,x):
        return [x['xmin'],x['ymin'],x['xmax'],x['ymax']],x['label']

    def __call__(self,raw_targets):
        targets={}
        results=list(map(self.parser,raw_targets))
        targets['bndboxes']=torch.tensor([result[0] for result in results])
        targets['labels']=torch.tensor([result[1] for result in results])
        return targets
        

transform_train=transforms.Compose([transforms.Resize([800,600],Image.BICUBIC),
                                    transforms.ColorJitter(brightness=0.5,
                                                           contrast=0.5,
                                                           saturation=0.5,
                                                           hue=0.5),
                                    transforms.ToTensor()])
transform_test=transforms.Compose([transforms.Resize([800,600],Image.BICUBIC),
                                   transforms.ToTensor()])
transform_target=TargetToTensor()
