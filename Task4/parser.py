import pandas as pd
import unicodedata
import re



class Parser(object):
    def __init__(self):
        super().__init__()
        self.sentences={}
        self.conversations=[]
        self.qa_pairs=[]
#        self.qa_pairs=pd.DataFrame(columns=['question','answer'])


    def load_sentences(self,file_name):
        fields=['lineID','characterID','movieID','character','text']

        with open(file_name,'r',encoding='iso-8859-1') as f:
            for line in f:
                line=line[:-1]
                line_items=line.split(' +++$+++ ')
                line_info={}

                for i,field in enumerate(fields):
                    line_info[field]=line_items[i]
                
                self.sentences[line_info['lineID']]=line_info


    def load_conversations(self,file_name):
        fields=['character1ID','character2ID','movieID','utteranceID']

        with open(file_name,'r',encoding='iso-8859-1') as f:
            for line in f:
                line=line[:-1]
                conversation=line.split(' +++$+++ ')
                line_info={}

                for i,field in enumerate(fields):
                    line_info[field]=conversation[i]

                lineID_list=eval(line_info['utteranceID'])
                line_info['utterance']=[]

                for utterance in lineID_list:
                    line_info['utterance'].append(
                                           self.sentences[utterance])

                self.conversations.append(line_info)


    def extract_conversation(self):
        for conversation in self.conversations:
            for i in range(len(conversation['utterance'])-1):
                qa={}
                qa['question']=\
                        conversation['utterance'][i]['text'].strip()
                qa['answer']=\
                        conversation['utterance'][i+1]['text'].strip()
                
                if qa['question'] != '' and qa['answer'] !='':
                    self.qa_pairs.append(qa)



class Dictionary(object):
    def __init__(self):
        super().__init__()
        self.word2index={'PAD':0,
                         'SOS':1,
                         'EOS':2}
        self.word_count={}
        self.num_words=3


    def add_sentence(self,sentence):
        for word in sentence.split(' '):
            self.add_word(word)


    def add_word(self,word):
        if word not in self.word2index.keys():
            self.word2index[word]=self.num_words
            self.word_count[word]=1
            self.num_words+=1
        else:
            self.word_count[word]+=1


    def reduce_dict(self,min_count):
        keep_words=[]

        for key,value in self.word_count.items():
            if value>min_count:
                keep_words.append(key)

        self.word_count={}
        self.word2index={'PAD':0,
                         'SOS':1,
                         'EOS':2}
        self.num_words=3

        for word in keep_words:
            self.add_word(word)



def unicode2ascii(word):
    return ''.join(c for c in unicodedata.normalize('NFD',word)
                   if unicodedata.category(c) !='Mn')


def normalize_str(s):
    s=unicode2ascii(s.lower().strip())
    s=re.sub(r'([.!?])',r' \1',s)
    s=re.sub(r'[^a-zA-Z.!?]+',r' ',s)
    s=re.sub(r'\s+',r' ',s).strip()
    return s


def check_pair_length(item1,item2,max_length):
    return len(item1.split(' '))<max_length and \
           len(item2.split(' '))<max_length


def filter_pairs(qa_pairs,max_length):
    dictionary=Dictionary()
    new_qa_pairs=[]

    for qa in qa_pairs:
        question=normalize_str(qa['question'])
        answer=normalize_str(qa['answer'])

        if check_pair_length(question,answer,max_length) == True:
            dictionary.add_sentence(question)
            dictionary.add_sentence(answer)
            new_qa_pairs.append({'question':question,
                                 'answer':answer})

    return new_qa_pairs,dictionary


def check_word_exists(dictionary,sentence):
    words=sentence.split(' ')

    for word in words:
        if word not in dictionary.word2index.keys():
            return False

    return True


def reduce_rare_words(dictionary,qa_pairs,min_count):
    dictionary.reduce_dict(min_count)
    new_qa_pairs=[]

    for qa in qa_pairs:
        question=qa['question']
        answer=qa['answer']

        if check_word_exists(dictionary,question)==True and \
           check_word_exists(dictionary,answer)==True:    
            new_qa_pairs.append({'question':question,'answer':answer})
    
    return new_qa_pairs


def sentence2int(sentence,dictionary):
    return [dictionary.word2index[w] for w in sentence.split(' ')]+[2]


def extent_qa_pairs(qa_pairs,dictionary,max_length):
    new_qa_pairs=[]

    for qa in qa_pairs:
        question=qa['question']
        answer=qa['answer']
        qa['question_int']=sentence2int(question,dictionary)
        qa['answer_int']=sentence2int(answer,dictionary)
        qa['question_len']=len(qa['question_int'])
        qa['answer_len']=len(qa['answer_int'])
        qa['question_int']=qa['question_int']+\
                           [0]*(max_length-qa['question_len'])
        qa['answer_int']=qa['answer_int']+\
                         [0]*(max_length-qa['answer_len'])
        qa['mask']=[1]*qa['answer_len']+[0]*(max_length-qa['answer_len'])
        new_qa_pairs.append(qa)

    return new_qa_pairs


#process raw data
parser=Parser()
parser.load_sentences('./Data/Cornell/movie_lines.txt')
parser.load_conversations('./Data/Cornell/movie_conversations.txt')
parser.extract_conversation()

#filter pairs
MAX_SENTENCE_LENGTH=10
qa_pairs,dictionary=filter_pairs(parser.qa_pairs,MAX_SENTENCE_LENGTH)

#delete rare word
MIN_COUNT=3
qa_pairs=reduce_rare_words(dictionary,qa_pairs,MIN_COUNT)

#padding and transfer str_list to int_list
qa_pairs=extent_qa_pairs(qa_pairs,dictionary,MAX_SENTENCE_LENGTH)

#save
dict_word=[]

for k,v in dictionary.word2index.items():
        dict_word.append({'word':k,'index':v})

dict_df=pd.DataFrame(columns=['word','index'])
dict_df=dict_df.append(dict_word,ignore_index=True)
dict_df.to_csv('./Data/Cornell_dictionary.csv',index=False)
qa_df=pd.DataFrame(columns=['question',
                            'answer',
                            'question_int',
                            'answer_int',
                            'question_len',
                            'answer_len',
                            'mask'])
qa_df=qa_df.append(qa_pairs,ignore_index=True)
qa_df.to_csv('./Data/Cornell_qa.csv',index=False)

