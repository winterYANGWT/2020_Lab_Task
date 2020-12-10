import torch


EPOCH_START=0
EPOCH=20
BATCH_SIZE=2
LEARNING_RATE=0.001
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
VOC_LABEL_NAME=('background',
                'aeroplane','bicycle','bird','boat','bottle',
                'bus','car','cat','chair','cow',
                'diningtable','dog','horse','motorbike','person',
                'pottedplant','sheep','sofa','train','tvmonitor')

