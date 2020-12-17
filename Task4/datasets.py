import config
import torch
import pandas as pd
import transforms



class Dataset(torch.utils.data.Dataset):
    def __init__(self,dict_csv,text_csv,transform=None):
        super().__init__()
        dict_frame=pd.read_csv(dict_csv)
        self.word2index=dict_frame.set_index('word')
        self.index2word=dict_frame.set_index('index')
        self.text_frame=pd.read_csv(text_csv)
        self.transform=transform
        self._num_word=len(self.word2index)


    @property
    def num_word(self):
        return self._num_word


    def __len__(self):
        return len(self.text_frame)


    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        line=self.text_frame.loc[index,:]
        question,answer=line[:2]
        question_int,answer_int=line[2:4]
        question_len,answer_len=[line[4]],[line[5]]
        mask=line[6]
        question_int=eval(question_int)
        answer_int=eval(answer_int)
        mask=eval(mask)
        
        data=question_int,\
             question_len,\
             answer_int,\
             answer_len,\
             mask

        if self.transform!=None:
            data=self.transform(data)

        return data



def collate_fn(batch):
    question_int=torch.stack([item[0] for item in batch]).permute(1,0)
    question_len=torch.cat([item[1] for item in batch])
    answer_int=torch.stack([item[2] for item in batch]).permute(1,0)
    answer_len=torch.cat([item[3] for item in batch])
    max_len=torch.max(answer_len)
    answer_int=answer_int[:max_len,:]
    mask=torch.stack([item[4] for item in batch]).permute(1,0)
    mask=mask[:max_len,:]
    return question_int,question_len,\
           answer_int,answer_len,mask


Cornell=Dataset('./Data/Cornell_dictionary.csv','./Data/Cornell_qa.csv',
                transform=transforms.ToTensor())

if __name__=='__main__':
    data_loader=torch.utils.data.DataLoader(dataset=Cornell,
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
