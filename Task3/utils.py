import os
import torch
import config


def load_model(model,load_path):
    model.load_state_dict(torch.load(load_path,map_location=config.DEVICE))

def save_model(model,save_path):
    torch.save(model.state_dict(),save_path,map_location=config.DEVICE)
