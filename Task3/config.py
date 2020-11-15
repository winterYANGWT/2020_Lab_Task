import torch


BATCH_SIZE=1
EPOCHES=20
LEARNING_RATE=0.0002
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
L1_LAMBDA=100
BETAS=[0.5,0.999]
