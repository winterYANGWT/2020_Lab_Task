import os
import torch
import config



def load_model(model,load_dir,name):
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)

    model.load_state_dict(torch.load(os.path.join(load_dir,name),
                                     map_location=config.DEVICE))


def save_model(model,save_dir,name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(),os.path.join(save_dir,name))


def tensor2list(tensor):
    '''
    tensor : B,Num_str,Max_len
    '''
    result=[]

    for b in tensor:
        str_list=[]

        for s in b:
            str_list.append(standard_list(s))

        result.append(str_list)

    return result


def standard_list(index_list):
    result=[]

    for index in index_list:
        if index==2:
            break
        elif index!=0:
            result.append(index.item())

    return result


def list_trim(l):
    new_l=[]

    for item in l:
        if item not in new_l:
            new_l.append(item)

    return new_l

