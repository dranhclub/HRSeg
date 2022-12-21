import torch
import torch.nn.functional as F
import os
from model import PolypSeg
import cv2
from dataloader import TestDatasets

TEST_SIZE = 352
TEST_ROOT = './dataset/TestDataset'
PTH_PATH = './model_pth/PolypSeg.e_120.Dec20.albumen.pth'

SAVE_ROOT_DIR = './result_map/albumen/'

# Prepare model
model = PolypSeg()
model.load_state_dict(torch.load(PTH_PATH))
model.cuda()
model.eval()

test_loader = TestDatasets(TEST_ROOT, 352)

for ds_name in test_loader.DS_NAMES:
    # Make save folder if not exist
    save_dir = os.path.join(SAVE_ROOT_DIR, ds_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_imgs = test_loader.datasets[ds_name]["n_imgs"]
    for i in range(n_imgs):
        # Get img, mask
        _, image, gt = test_loader.get_item(ds_name, i)
        image = image.unsqueeze(0).cuda()

        # Infer
        res = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        # Save
        img_filename = test_loader.datasets[ds_name]["imgs"][i]
        save_path = os.path.join(save_dir, img_filename)
        cv2.imwrite(save_path, res * 255)
    
    print("Done", ds_name)