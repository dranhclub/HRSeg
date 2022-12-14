import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import dice

DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

GT_PATH = './dataset/TestDataset'
# PRED_PATH = './result_map/PolypPVT' # result when inference 1 step
PRED_PATH = './result_map/twostep' # result when inferenec 2 step


img_idx = 0

all_ds_percents = []
all_ds_dice_scores = []

# For each dataset
for ds_idx, ds in enumerate(DS_NAMES):
    print("Dataset: ", ds)
    percents = []
    dice_scores = []

    # Get imgs of dataset
    img_dir = f"{GT_PATH}/{DS_NAMES[ds_idx]}/images"
    imgs = os.listdir(img_dir)
    
    # For each img
    for img_idx in range(len(imgs)):

        # Get GT of img
        gt_dir = f"{GT_PATH}/{DS_NAMES[ds_idx]}/masks"
        gt_filename = imgs[img_idx]
        gt_path = f"{GT_PATH}/{DS_NAMES[ds_idx]}/masks/{gt_filename}"

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

    print("Mean dice=", np.mean(dice_scores))

    # Plot histogram for this dataset
    plt.hist(percents, bins=30)
    plt.title("Dataset: " + ds)
    plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
    plt.ylabel("# Images")
    plt.show()

    # Plot scatter for this dataset
    plt.scatter(percents, dice_scores)
    plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
    plt.ylabel("Dice score (%)")
    plt.title("Dataset: " + ds)
    plt.show()

    all_ds_percents.extend(percents)
    all_ds_dice_scores.extend(dice_scores)

# Plot histogram for all dataset
plt.hist(all_ds_percents, bins=30)
plt.title("Dataset: all")
plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
plt.ylabel("# Images")
plt.show()

# Plot scatter for all dataset
plt.scatter(all_ds_percents, all_ds_dice_scores)
plt.xlabel("Percentage of [#white pixel]/[#pixel] (%)")
plt.ylabel("Dice score (%)")
plt.title("Dataset: all")
plt.show()