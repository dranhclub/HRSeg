import torch
import numpy as np
from thop import profile
from thop import clever_format
import os
from natsort import natsorted
import cv2
from prettytable import PrettyTable

# Constants
DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
TRAIN_DS_NAMES = ['TrainDataset', 'SunDataset', 'TrainDataset_synthesis']
TEST_ROOT = './dataset/TestDataset'
DATA_ROOT = './dataset'
RESULT_ROOT = './result_map'
CAPTURE_ROOT = './captured'

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

def dice(input, target):
    """
    input, target: gray image, normalized in [0,1]
    """
    SMOOTH = 1
    input_flat = np.reshape(input, (-1))
    target_flat = np.reshape(target, (-1))
    intersection = (input_flat * target_flat)
    dice = (2 * intersection.sum() + SMOOTH) / (input.sum() + target.sum() + SMOOTH)
    return dice

def polyp_size(gt):
    return gt[gt > 0].size / gt.size * 100

class RawDataset():
    def __init__(self, root, names) -> None:
        self.filenames = {}
        self.root = root
        self.names = names
        for ds_name in self.names:
            self.filenames[ds_name] = self.get_filenames(ds_name)

    def get_filenames(self, ds_name):
        img_dir = f"{self.root}/{ds_name}/images"
        filenames = natsorted(os.listdir(img_dir))
        return filenames

    def get_img_path(self, ds_name, img_idx):
        filename = self.filenames[ds_name][img_idx]
        return os.path.join(self.root, ds_name, "images", filename)
    
    def get_gt_path(self, ds_name, img_idx):
        filename = self.filenames[ds_name][img_idx]
        return os.path.join(self.root, ds_name, "masks", filename)

    def get_path(self, ds_name, img_idx):
        filename = self.filenames[ds_name][img_idx]
        img_path = self.get_img_path(ds_name, img_idx)
        gt_path = self.get_gt_path(ds_name, img_idx)
        return filename, img_path, gt_path
    
    def get_img(self, ds_name, img_idx):
        """Return BGR image"""
        path = self.get_img_path(ds_name, img_idx)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img

    def get_gt(self, ds_name, img_idx):
        """Return binary mask"""
        path = self.get_gt_path(ds_name, img_idx)
        gt = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return gt

    def get_random_img_gt(self):
        ds_name = np.random.choice(self.names)
        img_idx = np.random.choice(len(self.filenames[ds_name]))
        
        img = self.get_img(ds_name, img_idx)
        gt = self.get_gt(ds_name, img_idx)

        return img, gt

def get_train_raw_dataset():
    return RawDataset(DATA_ROOT, TRAIN_DS_NAMES)

def get_test_raw_dataset():
    return RawDataset(TEST_ROOT, DS_NAMES)

class RawResult():
    def __init__(self, name) -> None:
        self.name = name
        self.root = RESULT_ROOT
        self.filenames = {}
        for ds_name in DS_NAMES:
            self.filenames[ds_name] = self.get_filenames(ds_name)
    
    def get_filenames(self, ds_name):
        dir = f"{self.root}/{self.name}/{ds_name}"
        filenames = natsorted(os.listdir(dir))
        return filenames
    
    def get_result_path(self, ds_name, img_idx):
        path = os.path.join(self.root, self.name, ds_name, self.filenames[ds_name][img_idx])
        return path

    def get_result(self, ds_name, img_idx):
        """Return binary image"""
        path = self.get_result_path(ds_name, img_idx)
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def rotate(image, angle, center = None, scale = 1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params