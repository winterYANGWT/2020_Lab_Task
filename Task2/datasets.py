import config
import utils
import torch
import transforms
from PIL import Image
import pandas as pd


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 image_csv,anno_csv,
                 image_transform=None,bbox_transform=None,label_transform=None):
        super().__init__()
        self.image_data_frame=pd.read_csv(image_csv)
        self.anno_data_frame=pd.read_csv(anno_csv)
        self.image_transform=image_transform
        self.bbox_transform=bbox_transform
        self.label_transform=label_transform


    def __len__(self):
        return len(self.image_data_frame)


    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        image_path=self.image_data_frame.loc[index,'path']
        image=Image.open(image_path).convert('RGB')
        condition_filter=lambda:self.anno_data_frame.path==image_path
        annos=self.anno_data_frame.loc[condition_filter(),:].values.tolist()
        bboxes=[anno[1:5] for anno in annos]
        labels=[anno[5] for anno in annos]

        if self.image_transform!=None:
            image=self.image_transform(image)

        if self.bbox_transform!=None:
            bboxes=self.bbox_transform(bboxes)

        if self.label_transform!=None:
            labels=self.label_transform(labels)

        return image,bboxes,labels


def collate_fn(batch):
    images,targets=tuple(zip(*batch))
    images=torch.stack(images)
    targets=list(targets)
    return images,targets


@utils.variable
def VOC12_label_name():
    df=pd.read_csv('./Data/VOC12_dict.csv')
    df_list=df.values.tolist()
    l2n={item[0]:item[1] for item in df_list}
    n2l={item[1]:item[0] for item in df_list}
    return l2n,n2l


@utils.variable
def VOC12_train():
    return Dataset('./Data/VOC12_train_image.csv',
                   './Data/VOC12_train_anno.csv',
                   transforms.transform_img_train,
                   transforms.transform_bbox,
                   transforms.transform_label)

@utils.variable
def VOC12_val():
    return Dataset('./Data/VOC12_val_image.csv',
                   './Data/VOC12_val_anno.csv',
                   transforms.transform_test,
                   transforms.transform_bbox,
                   transforms.transform_label)


if __name__=='__main__':
    print(VOC12_train[0])

