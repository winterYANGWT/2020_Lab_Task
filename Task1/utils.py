import os
import torch
import config



def load_model(model,load_dir,name):
    if not os.path.exists(load_dir):
        os.makedirs(load_dir)

    model.load_state_dict(torch.load(os.path.join(load_dir,name)),map_location=config.DEVICE)


def save_model(model,save_dir,name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(),os.path.join(save_dir,name))

