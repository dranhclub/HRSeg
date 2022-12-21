import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import dice

DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

DS_PATH = './dataset/TestDataset'

PRED_PATH = './result_map/albumen'

img_idx = 0

all_ds_percents = {}
all_ds_dice_scores = {}

# For each dataset
for ds_idx, ds_name in enumerate(DS_NAMES):
    percents = []
    dice_scores = []

    # Get imgs of dataset
    img_dir = f"{DS_PATH}/{DS_NAMES[ds_idx]}/images"
    imgs = os.listdir(img_dir)
    
    # For each img
    for img_idx in range(len(imgs)):

        # Get GT of img
        gt_dir = f"{DS_PATH}/{DS_NAMES[ds_idx]}/masks"
        gt_filename = imgs[img_idx]
        gt_path = f"{DS_PATH}/{DS_NAMES[ds_idx]}/masks/{gt_filename}"

        # Get prediction
        pred_dir = f"{PRED_PATH}/{DS_NAMES[ds_idx]}"
        imgs = os.listdir(pred_dir)
        pred_filename = imgs[img_idx]
        pred_path = f"{PRED_PATH}/{DS_NAMES[ds_idx]}/{pred_filename}"

        # Read GT, pred
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Percentage of white pixel per total number of pixel
        percent = gt[gt > 0].size / gt.size * 100
        percents.append(percent)
        
        # Dice score
        dice_score = dice(gt / 255, pred / 255) * 100
        dice_scores.append(dice_score)

    mean_dice = np.mean(dice_scores)
    print(f"{ds_name}: {mean_dice:.2f} %")

    all_ds_percents[ds_name] = percents
    all_ds_dice_scores[ds_name] = dice_scores

plt.figure(figsize=(20, 11))
for i in range(len(all_ds_percents)):
    ds_name = DS_NAMES[i]
    percents = all_ds_percents[ds_name]
    dice_scores = all_ds_dice_scores[ds_name]

    # Plot histogram of polyp sizes for this dataset
    plt.subplot(2, 5, i + 1)
    plt.hist(percents, bins=50)
    plt.title("Dataset: " + ds_name)
    plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
    plt.ylabel("# Images")

    # Plot dice score versus polyp size for this dataset
    plt.subplot(2, 5, i + 1 + 5)
    plt.scatter(percents, dice_scores)
    plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
    plt.ylabel("Dice score (%)")
    plt.title("Dataset: " + ds_name)

plt.show()