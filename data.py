# data.py

from os.path import join
from os import listdir
from os.path import join
import random

from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
#from dataset import DatasetFromFolder

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # img = img.resize((256, 256), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir_input, image_dir_target, direction, image_size=[512, 512], patch_size=[256, 256]):
        super(DatasetFromFolder, self).__init__()
        self.direction = direction
        self.a_path = image_dir_input
        self.b_path = image_dir_target
        self.image_filenames = sorted([x for x in listdir(self.a_path) if is_image_file(x)])
        self.image_size = image_size
        self.patch_size = patch_size

        self.transform = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        a = Image.open(join(self.a_path, self.image_filenames[index])).convert('RGB')
        b = Image.open(join(self.b_path, self.image_filenames[index])).convert('RGB')
        # a = transforms.ToTensor()(a)
        # b = transforms.ToTensor()(b)
        
        a = self.transform(a)
        b = self.transform(b)
        
        # h_offset = random.randint(0, max(0, self.image_size[0] - self.patch_size[0] - 1))
        # w_offset = random.randint(0, max(0, self.image_size[1] - self.patch_size[1] - 1))

        # a = a[:, h_offset:h_offset + self.patch_size[0], w_offset:w_offset + self.patch_size[1]]
        # b = b[:, h_offset:h_offset + self.patch_size[0], w_offset:w_offset + self.patch_size[1]]

        # a = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(a)
        # b = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(b)

        # # hori/verti flip
        # if random.random() < 0.5:
        #     idx = [i for i in range(a.size(2) - 1, -1, -1)]
        #     idx = torch.LongTensor(idx)
        #     a = a.index_select(2, idx)
        #     b = b.index_select(2, idx)

        if self.direction == "a2b":
            return a, b
        else:
            return b, a

    def __len__(self):
        return len(self.image_filenames)

def get_training_set(train_dataset_input_dir, train_dataset_target_dir, direction, image_size, patch_size):
    return DatasetFromFolder(train_dataset_input_dir, train_dataset_target_dir, direction, image_size, patch_size)


def get_test_set(test_dataset_input_dir, test_dataset_target_dir, direction, image_size, patch_size):
    return DatasetFromFolder(test_dataset_input_dir, test_dataset_target_dir, direction, image_size, patch_size)
