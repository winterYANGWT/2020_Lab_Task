import os
import pandas as pd



class Parser(object):
    def __init__(self):
        self.caption_list=[]
        self.image_count=0
        self.max_len=0


    def read_line(self,caption_file_path,image_folder_path):
        with open(caption_file_path,'r') as f:
            f.readline()

            for line in f.readlines():
                line=line[:-1]
                split_index=line.find('.jpg')+4
                image_name=line[:split_index]
                caption=line[split_index+1:]

                if caption[0]=='\"' and caption[-1]=='\"':
                    caption=caption[1:-2]

                info_dict={}
                info_dict['caption']=caption.lower()
                info_dict['image_name']=image_name
                self.caption_list.append(info_dict)


    def get_info(self):
        for d in self.caption_list:
            d['str_list']=d['caption'].split(' ')
            d['len']=len(d['str_list'])

            if d['len']>self.max_len:
                self.max_len=d['len']

class VOC(object):
    def __init__(self):
        self.word2index={'PAD':0,
                         'SOS':1,
                         'EOS':2}
        self.index2word={0:'PAD',
                         1:'SOS',
                         2:'EOS'}
        self.num_words=3


    def add_word(self,word):
        if word not in self.word2index.keys():
            self.word2index[word]=self.num_words
            self.index2word[self.num_words]=word
            self.num_words+=1



def word2int(parser,voc):
    for d in parser.caption_list:
        str_list=d['str_list']
        d['int_list']=[voc.word2index[word] for word in str_list]


def pad_zero(parser):
    for d in parser.caption_list:
        d['int_list']+=[2]
        d['mask']=[1 for _ in d['int_list']]
        d['int_list']+=[0]*(parser.max_len-d['len'])
        d['mask']+=[0]*(parser.max_len-d['len'])
        d['len']+=1


def process(parser,voc,image_floder_path,parser_path,voc_path):
    caption_df=pd.DataFrame(columns=['image_path',
                                     'caption',
                                     'int_list',
                                     'len',
                                     'mask'])
    dictionary_df=pd.DataFrame(columns=['word','index'])
    dictionary_list=[]

    for d in parser.caption_list:
        d['image_path']=os.path.join(image_floder_path,d['image_name'])
        d.pop('str_list')
        d.pop('image_name')

    for key in voc.word2index.keys():
        info_dict={}
        info_dict['word']=key
        info_dict['index']=voc.word2index[key]
        dictionary_list.append(info_dict)

    caption_df=caption_df.append(parser.caption_list,ignore_index=True)
    dictionary_df=dictionary_df.append(dictionary_list,ignore_index=True)
    caption_df.to_csv(parser_path,index=False)
    dictionary_df.to_csv(voc_path,index=False)


if __name__=='__main__':
    parser=Parser()
    parser.read_line('./Data/Flickr8k/captions.txt','./Data/Flickr8k/Images')
    parser.get_info()
    voc=VOC()

    for d in parser.caption_list:
        for word in d['str_list']:
            voc.add_word(word)

    word2int(parser,voc)
    pad_zero(parser)
    process(parser,voc,
            './Data/Flickr8k/Images',
            './Data/Flickr8k_caption.csv',
            './Data/Flickr8k_dictionary.csv')

