import torch



BATCH_SIZE=64
LR=0.0001
EPOCH_START=0
LOAD=EPOCH_START!=0
EPOCH=400
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TEACHER_FORCING_RATIO=0.5
CLIP=50.0

