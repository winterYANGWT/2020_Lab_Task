import os
import pandas as pd



class Parser(object):
    def __init__(self):
        self.image_dict={}
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

                if image_name not in self.image_dict.keys():
                    self.image_dict[image_name]=[]
                    self.image_count+=1

                caption_dict={}
                caption_dict['caption']=caption.lower()
                self.image_dict[image_name].append(caption_dict)


    def get_info(self):
        for key in self.image_dict.keys():
            caption_list=[]
            caption_len_list=[]

            for caption_dict in self.image_dict[key]:
                caption_dict['str_list']=caption_dict['caption'].split(' ')
                caption_dict['len']=len(caption_dict['str_list'])

                if caption_dict['len']>self.max_len:
                    self.max_len=caption_dict['len']

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
        else:
            self.num_words+=1



def word2int(parser,voc):
    for key in parser.image_dict.keys():
        for caption_dict in parser.image_dict[key]:
            str_list=caption_dict['str_list']
            caption_dict['int_list']=[voc.word2index[word] for word in str_list]


def pad_zero(parser):
    for key in parser.image_dict.keys():
        for caption_dict in parser.image_dict[key]:
            caption_dict['int_list']+=[2]
            caption_dict['mask']=[1 for _ in caption_dict['int_list']]
            caption_dict['int_list']+=[0]*(parser.max_len-caption_dict['len'])
            caption_dict['mask']+=[0]*(parser.max_len-caption_dict['len'])
            caption_dict['len']+=1


def process(parser,voc,image_floder_path,parser_path,voc_path):
    caption_df=pd.DataFrame(columns=['image_path',
                                     'caption0',
                                     'caption0_int',
                                     'caption0_len',
                                     'caption0_mask',
                                     'caption1',
                                     'caption1_int',
                                     'caption1_len',
                                     'caption1_mask',
                                     'caption2',
                                     'caption2_int',
                                     'caption2_len',
                                     'caption2_mask',
                                     'caption3',
                                     'caption3_int',
                                     'caption3_len',
                                     'caption3_mask',
                                     'caption4',
                                     'caption4_int',
                                     'caption4_len',
                                     'caption4_mask'])
    dictionary_df=pd.DataFrame(columns=['word','index'])
    caption_list=[]
    dictionary_list=[]

    for key in parser.image_dict.keys():
        info_dict={}
        info_dict['image_path']=os.path.join(image_floder_path,key)
        
        for i,caption_dict in enumerate(parser.image_dict[key]):
            info_dict['caption'+str(i)]=caption_dict['caption']
            info_dict['caption'+str(i)+'_int']=caption_dict['int_list']
            info_dict['caption'+str(i)+'_len']=caption_dict['len']
            info_dict['caption'+str(i)+'_mask']=caption_dict['mask']

        caption_list.append(info_dict)

    for key in voc.word2index.keys():
        info_dict={}
        info_dict['word']=key
        info_dict['index']=voc.word2index[key]
        dictionary_list.append(info_dict)

    caption_df=caption_df.append(caption_list,ignore_index=True)
    dictionary_df=dictionary_df.append(dictionary_list,ignore_index=True)
    caption_df.to_csv(parser_path,index=False)
    dictionary_df.to_csv(voc_path,index=False)


if __name__=='__main__':
    parser=Parser()
    parser.read_line('./Data/Flickr8k/captions.txt','./Data/Flickr8k/Images')
    parser.get_info()
    voc=VOC()

    for key in parser.image_dict.keys():
        for caption_dict in parser.image_dict[key]:
            for word in caption_dict['str_list']:
                voc.add_word(word)

    word2int(parser,voc)
    pad_zero(parser)
    process(parser,voc,
            './Data/Flickr8k',
            './Data/Flickr8k_caption.csv',
            './Data/Flickr8k_dictionary.csv')

