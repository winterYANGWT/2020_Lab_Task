import config
import torch
import pandas as pd



class Dataset(torch.utils.data.Dataset):
    def __init__(self,dict_csv,text_csv,transform=None):
        super().__init__()
        dict_frame=pd.read_csv(dict_csv)
        self.word2index=dict_frame.set_index('word')
        self.index2word=dict_frame.set_index('index')
        self.text_frame=pd.read_csv(text_csv)
        self.transform=transform


    def __len__(self):
        return len(self.text_frame)


    def __getitem__(self,index):
        if torch.is_tensor(index)==True:
            index=index.tolist()

        line=self.text_frame.loc[index,:]
        question,answer=line[:2]
        question_int,answer_int=line[2:]
        question_int=eval(question_int)
        answer_int=eval(answer_int)

        if self.transform!=None:
            question=self.transform(question)
            answer=self.transform(answer)

        return question_int,answer_int



Cornell=Dataset('./Data/Cornell_dictionary.csv','./Data/Cornell_qa.csv')

if __name__=='__main__':
    print(Cornell[3])

