import torch
import torch.nn.functional as F
import os
from model import HRSeg
import cv2
from dataloader import TestDatasets
from utils import TEST_ROOT, RESULT_ROOT
import shutil
from train import infer

TEST_SIZE = 576
PTH_PATH = './model_pth/HRSeg.e_40.Jan31-04h00.pth'

name = 'HRSeg'
print("NAME=", name)
save_dir = os.path.join(RESULT_ROOT, name)
print("save_dir=", save_dir)

shutil.rmtree(save_dir, ignore_errors=True)
os.makedirs(save_dir, exist_ok=True)

# Prepare model
model = HRSeg().cuda()
model.load_state_dict(torch.load(PTH_PATH, map_location='cuda'))
model.eval()

test_loader = TestDatasets(TEST_ROOT, TEST_SIZE)

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

        # Save
        img_filename = test_loader.datasets[ds_name]["imgs"][i]
        save_path = os.path.join(save_sub_dir, img_filename)
        cv2.imwrite(save_path, res * 255)
    
    print("Done", ds_name)