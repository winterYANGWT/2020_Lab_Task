import torch
import torch.utils.data as data
import pandas as pd
import transforms
from PIL import Image



class Dataset(data.Dataset):
    def __init__(self,caption_csv,dict_csv,image_transform=None,text_transform=None):
        super().__init__()
        self.caption_frame=pd.read_csv(caption_csv)
        dict_frame=pd.read_csv(dict_csv)
        self.word2index=dict_frame.set_index('word')
        self.index2word=dict_frame.set_index('index')
        self.image_transform=image_transform
        self.text_transform=text_transform
        self._num_word=len(self.word2index)


    @property
    def num_word(self):
        return self._num_word


    def __len__(self):
        return len(self.caption_frame)


    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        caption_info=self.caption_frame.loc[index,:]
        image_path=caption_info[0]
        image=Image.open(image_path).convert('RGB')
        caption=caption_info[1]
        caption_int=eval(caption_info[2])
        caption_len=caption_info[3]
        caption_mask=eval(caption_info[4])
        
        if self.image_transform!=None:
            image=self.image_transform(image)

        if self.text_transform!=None:
            caption_int,caption_len,caption_mask=self.text_transform(caption_int,
                                                                     caption_len,
                                                                     caption_mask)

        return image,caption_int,caption_len,caption_mask



def collate_fn(batch):
    image=torch.stack([item[0] for item in batch])
    caption_int=torch.stack([item[1] for item in batch]).permute(1,0)
    caption_len=torch.cat([item[2] for item in batch])
    max_len=torch.max(caption_len)
    caption_mask=torch.stack([item[3] for item in batch]).permute(1,0)
    caption_int=caption_int[:max_len,:]
    caption_mask=caption_mask[:max_len,:]
    return image,caption_int,max_len,caption_mask


Flickr8k=Dataset('./Data/Flickr8k_caption.csv',
                 './Data/Flickr8k_dictionary.csv',
                 transforms.image_transform,
                 transforms.text_transform)

if __name__=='__main__':
    data_loader=torch.utils.data.DataLoader(dataset=Flickr8k,
                                            batch_size=5,
                                            collate_fn=collate_fn,
                                            shuffle=False)

    for i in data_loader:
        for j in i:
            print(j)

        print('')
        print('')
        print('')
        print('')
        print('')
