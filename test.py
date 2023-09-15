# test.py

# from __future__ import print_function
import os
import argparse
from tqdm import tqdm
from datetime import datetime

import torch
import torchvision.transforms as transforms

from data import is_image_file, load_img, save_img

# Testing settings
now = datetime.now()
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
parser.add_argument('--project_dir', type=str, default='', help='project directory')
parser.add_argument('--outputs_dir', type=str, default='outputs', help='experiments path in root dir')
parser.add_argument('--inference_instance_folder_name', type=str, default=str(now.strftime("%d_%m_%Y__%H%M%S"), default='folder name for current inferencing instance'))
parser.add_argument('--model_weights_path', type=str, default='', help='path to trained model weights')

parser.add_argument('--test_dataset_input_dir', type=str, default='', help='directory to test dataset: from')
parser.add_argument('--test_dataset_target_dir', type=str, default='', help='directory to test dataset: to')
parser.add_argument('--image_size', default=[512, 512], nargs='+', help='resize image to [h, w]')

parser.add_argument('--dataset', type=str, default='facades', help='facades')
parser.add_argument('--direction', type=str, default='a2b', help="a2b or b2a. b2a inverts dataset's input and target")
parser.add_argument('--cuda', action='store_true', help='use cuda')
parser.add_argument('--gpu_id', type=int, default=0, gelp='ID odf GPU to use')
#opt = parser.parse_args()
opt, unknown = parser.parse_known_args()

print(opt)

device = torch.device(f"cuda:{opt.gpu_id}" if opt.cuda else "cpu")

print("===> Loading Model")
net_g = torch.load(opt.model_weights_path).to(device)

if opt.direction == "a2b":
    image_dir = opt.test_dataset_input_dir
else:
    image_dir = opt.test_dataset_target_dir

image_filenames = sorted([x for x in os.listdir(image_dir) if is_image_file(x)])

transform = transforms.Compose([
        transforms.Resize((opt.image_size[0], opt.image_size[1])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
])

print("===> Starting Inferencing")
for image_name in tqdm(image_filenames, total=len(image_filenames)):
    img = load_img(image_dir + image_name)
    img = transform(img)
    input = img.unsqueeze(0).to(device)
    out = net_g(input)
    out_img = out.detach().squeeze(0).cpu()

    if not os.path.exists(os.path.join(opt.project_dir, opt.outputs_dir, opt.inference_instance_folder_name)):
        os.makedirs(os.path.join(opt.project_dir, opt.outputs_dir, opt.inference_instance_folder_name))
    save_img(out_img, os.path.join(opt.project_dir, opt.outputs_dir, opt.inference_instance_folder_name, image_name))
print("ALL DONE")