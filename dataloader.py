import torch.utils.data as data

import os
import torch
from torchvision import transforms

import cv2
import random
import numpy as np


class ColoringTransform(object):
    def __init__(self, size=256, mode="training"):
        super(ColoringTransform, self).__init__()
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([transforms.ToTensor()])

    def bgr_to_lab(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, ab = lab[:, :, 0], lab[:, :, 1:]
        return l, ab

    def gray_cue(self, bgr, threshold=0.99):
        h, w, c = bgr.shape
        cue = np.random.random([h, w, 1]) > threshold
        return cue

    def img_to_mask(self, cue_img):
        cue = cue_img[:, :, 0, np.newaxis] > 0 
        return cue

    def __call__(self, img, cue_img=None):
        threshold = 0.99
        if (self.mode == "training") | (self.mode == "validation"):
            image = cv2.resize(img, (self.size, self.size))
            cue = self.gray_cue(image, threshold)

            gray_image = image * cue

            l, ab = self.bgr_to_lab(image)
            l_gray, ab_gray = self.bgr_to_lab(gray_image)

            return self.transform(l), self.transform(ab), self.transform(ab_gray), self.transform(cue)

        elif self.mode == "testing":
            image = cv2.resize(img, (self.size, self.size))
            image = img
            cue = self.img_to_mask(cue_img)
            gray_image = image * cue

            l, _ = self.bgr_to_lab(image)
            _, ab_gray = self.bgr_to_lab(gray_image)

            return self.transform(l), self.transform(ab_gray), self.transform(cue)

        else:
            return NotImplementedError


class ColoringDataset(data.Dataset):
    def __init__(self, data_dir, size):
        super(ColoringDataset, self).__init__()

        self.data_dir = data_dir
        self.size = size
        self.transforms = None
        self.samples = None
        self.gray = None
        self.cue = None

    def set_mode(self, mode):
        self.mode = mode
        self.transforms = ColoringTransform(self.size, mode)
        if mode == "training":
            train_dir = os.path.join(self.data_dir, "train")
            self.samples = [os.path.join(self.data_dir, "train", img_item) for img_item in os.listdir(train_dir)]
        elif mode == "validation":
            val_dir = os.path.join(self.data_dir, "validation")
            self.samples = [os.path.join(self.data_dir, "validation", img_item) for img_item in os.listdir(val_dir)]
        elif mode == "testing":
            gray_dir = os.path.join(self.data_dir, "gray")
            cue_dir = os.path.join(self.data_dir, "cue")
            self.gray = [os.path.join(self.data_dir, "gray", img_item) for img_item in os.listdir(gray_dir)]
            self.cue = [os.path.join(self.data_dir, "cue", img_item) for img_item in os.listdir(cue_dir)]
        else:
            raise NotImplementedError

    def __len__(self):
        if self.mode != "testing":
            return len(self.samples)
        else:
            return len(self.gray)

    def __getitem__(self, idx):
        if self.mode == "testing":
            gray_file_name = self.gray[idx]
            cue_file_name = self.cue[idx]
            gray_img = cv2.imread(gray_file_name)
            cue_img = cv2.imread(cue_file_name)

            input_l, input_hint, cue = self.transforms(gray_img, cue_img)
            sample = {"l": input_l, "gray": input_hint, "cue": cue,
                      "file_name": "image_%06d.png" % int(os.path.basename(gray_file_name).split('.')[0])}
        else:
            file_name = self.samples[idx]
            img = cv2.imread(file_name)
            l, ab, gray, cue = self.transforms(img)
            sample = {"l": l, "ab": ab, "gray": gray, "cue": cue}

        return sample


def tensor2img(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.clip((np.transpose(image_numpy, (1, 2, 0))), 0, 1) * 255.0
    return image_numpy.astype(imtype)
