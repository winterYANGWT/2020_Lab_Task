import config
import transforms
import torch
import pandas as pd
from PIL import Image


class Dataset(torch.utils.data.Dataset):
    def __init__(self,csv_file,transform=None):
        super().__init__()
        self.data_frame=pd.read_csv(csv_file)
        self.transform=transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        input_path,target_path=self.data_frame.loc[index,:]
        input_image=Image.open(input_path).convert('RGB')
        target_image=Image.open(target_path).convert('RGB')

        if self.transform!=None:
            input_image,target_image=self.transform(input_image,target_image)

        return input_image.to(config.DEVICE),target_image.to(config.DEVICE)


maps_train=Dataset('./Data/maps_train.csv',transform=transforms.transform_train)
maps_test=Dataset('./Data/maps_val.csv',transform=transforms.transform_test)


if __name__=='__main__':
    print(maps_train[1])    
