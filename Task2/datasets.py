import config
import torch
import transforms
from PIL import Image
import pandas as pd


class VOC12(torch.utils.data.Dataset):
    def __init__(self,
                 image_csv,
                 annotation_csv,
                 image_transform=None,
                 target_transform=None):
        super().__init__()
        self.image_data_frame=pd.read_csv(image_csv)
        self.annotation_data_frame=pd.read_csv(annotation_csv)
        self.image_transform=image_transform
        self.target_transform=target_transform

    def __len__(self):
        return len(self.image_data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        image_path=self.image_data_frame.loc[index,'path']
        image=Image.open(image_path).convert('RGB')
        targets=self.annotation_data_frame.loc[self.annotation_data_frame['path']==image_path,:].to_dict('records')

        if self.image_transform!=None:
            image=self.image_transform(image)

        if self.target_transform!=None:
            targets=self.target_transform(targets)
        
        return image,targets


def collate_fn(batch):
    images,targets=tuple(zip(*batch))
    images=torch.stack(images)
    targets=list(targets)
    return images,targets

VOC_LABEL_NAME=('background',
                'aeroplane','bicycle','bird','boat','bottle',
                'bus','car','cat','chair','cow',
                'diningtable','dog','horse','motorbike','person',
                'pottedplant','sheep','sofa','train','tvmonitor')

VOC12_train=VOC12('./Data/VOC12_train_image.csv',
                  './Data/VOC12_train_anno.csv',
                  transforms.transform_train,
                  transforms.transform_target)

VOC12_val=VOC12('./Data/VOC12_val_image.csv',
                './Data/VOC12_val_anno.csv',
                transforms.transform_test,
                transforms.transform_target)

if __name__=='__main__':
    print(VOC12_train[0])
