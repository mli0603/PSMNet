import os
import os.path
import random

import numpy as np
import torch.utils.data as data
from PIL import Image
from natsort import natsorted
from . import readpfm as rp
import math
from dataloader import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    left_data = [os.path.join(filepath, obj, 'im0.png') for obj in os.listdir(filepath)]
    right_data = [os.path.join(filepath, obj, 'im1.png') for obj in os.listdir(filepath)]
    disp_left_data = [os.path.join(filepath, obj, 'disp0GT.pfm') for obj in os.listdir(filepath)]

    left_data = natsorted(left_data)
    right_data = natsorted(right_data)
    disp_left_data = natsorted(disp_left_data)

    return left_data, right_data, disp_left_data, left_data, right_data, disp_left_data


def default_loader(path):
    return Image.open(path).convert('RGB')


def disparity_loader(path):
    return rp.readPFM(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = natsorted(left)
        self.right = natsorted(right)
        self.disp_L = natsorted(left_disparity)
        directory = '/data/Data/ETH3D/two_view_training'
        self.occ_data = [os.path.join(directory, obj, 'mask0nocc.png') for obj in os.listdir(directory)]
        self.occ_data = natsorted(self.occ_data)

        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]
        left_img = self.loader(left)

        right_img = self.loader(right)
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)
        occL = np.array(Image.open(self.occ_data[index])) != 255
        dataL[occL] = 0.0

        if self.training:
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size
            th, tw = math.ceil(h / 16) * 16, math.ceil(w / 16) * 16

            left_img = np.pad(left_img, ((th - h, 0), (tw - w, 0), (0, 0)), 'constant', constant_values=0)
            right_img = np.pad(right_img, ((th - h, 0), (tw - w, 0), (0, 0)), 'constant', constant_values=0)
            dataL = np.pad(dataL, ((th - h, 0), (tw - w, 0)), 'constant', constant_values=0)

            processed = preprocess.get_transform(augment=False)
            left_img = processed(left_img)
            right_img = processed(right_img)
            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
