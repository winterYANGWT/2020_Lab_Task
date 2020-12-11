import torch



class ToTensor(object):
    def __init__(self):
        super().__init__()


    def __call__(self,data):
        question_int=torch.LongTensor(data[0])
        question_len=torch.ByteTensor(data[1])
        answer_int=torch.LongTensor(data[2])
        answer_len=torch.ByteTensor(data[3])
        mask=torch.ByteTensor(data[4])
        return question_int,question_len,\
               answer_int,answer_len,mask