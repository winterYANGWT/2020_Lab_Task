import torch
import torch.utils.data as data
import pandas 
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
        image=Image.open('image_path').convert('RGB')
        caption=caption_info[1]
        caption_int=caption_info[2]
        caption_len=caption_info[3]
        caption_mask=caption_info[4]
        
        if self.image_transform!=None:
            image=self.image_transform(image)

        if self.text_transform!=None:
            caption_int,caption_len,caption_mask=self.text_transfrom(caption_int,
                                                                     caption_len,
                                                                     caption_mask)

        return image,caption_int,caption_len,caption_mask

