import torch
import numpy as np
from thop import profile
from thop import clever_format
import os
from natsort import natsorted

# Constants
DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
TEST_ROOT = './dataset/TestDataset'
TRAIN_ROOT = './dataset/TrainDataset'
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


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


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


def get_filenames(ds_name):
    img_dir = f"{TEST_ROOT}/{ds_name}/images"
    imgs = natsorted(os.listdir(img_dir))

    return imgs

def get_test_img_gt_path(ds_name, img_idx):
    imgs = get_filenames(ds_name)

    # Get filename
    img_filename = imgs[img_idx]

    # Get img
    img_path = f"{TEST_ROOT}/{ds_name}/images/{img_filename}"

    # Get gt
    gt_path = f"{TEST_ROOT}/{ds_name}/masks/{img_filename}"

    return img_filename, img_path, gt_path

def get_test_result_path(name, ds_name, img_idx):
    result_dir = os.path.join(RESULT_ROOT, name)
    imgs = get_filenames(ds_name)

    # Get prediction path
    pred_filename = imgs[img_idx]
    ret = f"{result_dir}/{ds_name}/{pred_filename}"

    return ret