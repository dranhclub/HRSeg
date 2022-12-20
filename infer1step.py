import torch
import torch.nn.functional as F
import numpy as np
import os
from model import PolypSeg
import cv2
from . import utils
from torchvision import transforms

TEST_SIZE = 400
PTH_PATH = './model_pth/PolypSeg.e_120.07h59.pth'
# PTH_PATH = './model_pth/Affine big only e120.pth'

SAVE_ROOT_DIR = './result_map/testsize400/'

# Prepare model
model = PolypSeg()
model.load_state_dict(torch.load(PTH_PATH))
model.cuda()
model.eval()

# Prepare transformation
transform = transforms.Compose([
    transforms.Resize((TEST_SIZE, TEST_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])])

def infer(PIL_Img):
    """
    image: PIL image.
    res: mask as a numpy array with value in range 0-1.
    """
    input = transform(PIL_Img).unsqueeze(0).cuda()
    res = model(input)
    width, height = PIL_Img.size[0], PIL_Img.size[1]
    res = F.upsample(res, size=(height, width), mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return res

# For each dataset
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path = './dataset/TestDataset/{}'.format(_data_name)
    save_dir = os.path.join(SAVE_ROOT_DIR, _data_name)

    # Make save folder if not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    
    imgs = os.listdir(image_root)

    for img_idx in range(len(imgs)):
        # Get image
        img_filename = imgs[img_idx]
        img_path = f"{image_root}/{img_filename}"
        image = utils.rgb_loader(img_path)

        # Get gt
        gt_filename = imgs[img_idx]
        gt_path = f"{gt_root}/{gt_filename}"
        gt = utils.binary_loader(gt_path)

        # Predict
        res = infer(image)

        # Save
        save_path = os.path.join(save_dir, img_filename)
        cv2.imwrite(save_path, res * 255)

    print(_data_name, 'Finish!')
