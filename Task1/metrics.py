import torch
import torch.nn as nn
from torchtext.data.utils import ngrams_iterator
import numpy as np
import math
import collections
import utils



class LossMeter(object):
    def __init__(self):
        super().__init__()
        self.reset()


    def reset(self):
        self._value=0
        self.sum=0
        self.count=0


    def update(self,loss,count):
        self.count+=count
        self.sum+=count*loss
        self._value=self.sum/self.count

    
    @property
    def value(self):
        return self._value



class BLEUMeter(object):
    def __init__(self,max_n,mode='mean'):
        super().__init__()
        self.n=max_n
        
        if mode not in ['mean','single']:
            raise ValueError(mode,'should be mean or single')
        
        if mode=='mean':
            weight=[1/self.n]*self.n
        else:
            weight=[0]*self.n
            weight[self.n-1]=1

        self.weight=torch.tensor(weight)
        self.reset()


    def reset(self):
        self._value=0
        self.sum=0
        self.count=0


    def update(self,candidates_batch,references_batch):
        assert len(candidates_batch)==len(references_batch),\
                '''
                The number of batches of candidates and references should be the same 
                '''

        for candidates,references in zip(candidates_batch,references_batch):
            self.count+=1
            self.sum+=self.calc_bleu(candidates,references)

        self._value=self.sum/self.count


    def calc_bleu(self,candidates,references):
        result=0
        clip_count=torch.zeros(self.n)
        count=torch.zeros(self.n)

        r_count=collections.Counter()
        c_count=collections.Counter()
        c_clip_count=collections.Counter()

        for r in references:
            r_count=r_count|self.calc_count(r,self.n)

        for c in candidates:
            count=self.calc_count(c,self.n)
            c_count+=count
            c_clip_count+=count&r_count

        for ngram in c_clip_count:
            clip_count[len(ngram)-1]=c_clip_count[ngram]

        for ngram in c_count:
            count[len(ngram)-1]=c_count[ngram]

        if min(clip_count)==0:
            return 0
        else:
            pn=clip_count/count
            score=torch.exp(torch.sum(self.weight*torch.log(pn)))
            bp=self.calc_bp(candidates,references)
            return bp*score.item()


    def calc_count(self,tokens,max_n):
        assert max_n>0
        tokens=[str(token) for token in tokens]
        ngrams_counter=collections.Counter(tuple(x.split(' '))\
                                           for x in ngrams_iterator(tokens,max_n))
        return ngrams_counter


    def calc_bp(self,candidates,references):
        c_total_len=0
        r_total_len=0
        r_len_list=[len(r) for r in references]

        for c in candidates:
            c_len=len(c)
            c_total_len+=c_len
            r_total_len+=min(r_len_list,key=lambda x:abs(c_len-x))

        bp=math.exp(min(1-r_total_len/c_total_len,0))
        return bp

    
    @property
    def value(self):
        return self._value

