import argparse
import os
import shutil

import cv2
import torch
import torch.nn.functional as F

from dataloader import TestDatasets
from model import HRSeg
from train import infer
from utils import OUTER_SIZE, RESULT_ROOT, TEST_ROOT

import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--name", "-n", type=str, required=True)
    parser.add_argument("--pth_path", "-p", type=str, required=True)
    opt = parser.parse_args()

    # Prepare model
    model = HRSeg().cuda()
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda'))
    model.eval()

    print("NAME=", opt.name)
    save_dir = os.path.join(RESULT_ROOT, opt.name)
    print("save_dir=", save_dir)

    shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)
    print("Make dir:", save_dir)

    test_loader = TestDatasets(TEST_ROOT, OUTER_SIZE)

    for ds_name in test_loader.DS_NAMES:
        # Make save folder if not exist
        save_sub_dir = os.path.join(save_dir, ds_name)
        os.makedirs(save_sub_dir, exist_ok=True)

        n_imgs = test_loader.datasets[ds_name]["n_imgs"]
        for i in range(n_imgs):
            # Get img, mask
            _, image, gt = test_loader.get_item(ds_name, i)

            # Infer
            res = infer(model, image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            res = (res > 0.5).astype('float')

            # Save
            img_filename = test_loader.datasets[ds_name]["imgs"][i]
            save_path = os.path.join(save_sub_dir, img_filename)
            cv2.imwrite(save_path, res * 255)
        
        print("Done", ds_name)