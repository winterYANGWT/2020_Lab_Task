import argparse
import utils
import models
import torch
import transforms
from PIL import Image

parser=argparse.ArgumentParser()
parser.add_argument('--image-path',type=str,required=True)
parser.add_argument('--output-path',type=str,required=True)
parser.add_argument('--model-path',type=str,required=True)
args=parser.parse_args()

model=models.UNet()
#utils.load_model(model,args.model_path[:-5],args.model_path[-5:])

img_rgb=Image.open(args.image_path).convert('RGB')
img_tensor=transforms.transform_rgb2tensor(img_rgb)
img_tensor=torch.unsqueeze(img_tensor,0)

with torch.no_grad():
    fake_img_tensor=model(img_tensor)
    fake_img_tensor=torch.squeeze(fake_img_tensor,0)
    print(fake_img_tensor)

fake_img_rgb=transforms.transform_tensor2rgb(fake_img_tensor)
fake_img_rgb.save(args.output_path)
