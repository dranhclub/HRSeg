# Statistic and analysis results

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import dice, get_test_img_gt_path, get_test_result_path, get_filenames
from visualize import Visualizer
from utils import DS_NAMES

NAME = "spatter_noise"

def dice_vs_size():
    result = {
        'CVC-300': {
            'size': [],
            'dice': []
        }, 
        'CVC-ClinicDB': {
            'size': [],
            'dice': []
        }, 
        'Kvasir': {
            'size': [],
            'dice': []
        }, 
        'CVC-ColonDB': {
            'size': [],
            'dice': []
        }, 
        'ETIS-LaribPolypDB': {
            'size': [],
            'dice': []
        },
    }

    # For each dataset
    for ds_name in DS_NAMES:
        print("Processing", ds_name)

        imgs = get_filenames(ds_name)        

        # For each img
        for img_idx in range(len(imgs)):

            filename, img_path, gt_path = get_test_img_gt_path(ds_name, img_idx)
            pred_path = get_test_result_path(NAME, ds_name, img_idx)

            # Read GT, pred
            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            # Percentage of white pixel per total number of pixel
            percent = gt[gt > 0].size / gt.size * 100
            result[ds_name]['size'].append(percent)
            
            # Dice score
            dice_score = dice(gt / 255, pred / 255) * 100
            result[ds_name]['dice'].append(dice_score)

    print("Process done!")
    return result


result = dice_vs_size()


def combine(result):
    new_result = result.copy()
    new_result["all"] = {
        "size": [],
        "dice": []
    }
    for ds_name, value in result.items():
        new_result["all"]["size"].extend(value["size"])
        new_result["all"]["dice"].extend(value["dice"])
    return new_result


result_combined = combine(result)

################### Plot & Visualize ###################
visualizer = Visualizer(NAME)

def onpick(event):
    hit_list = event.ind
    img_idx = hit_list[0]
    ds_name = event.artist.axes.get_title()
    ds_idx = DS_NAMES.index(ds_name)
    print(ds_name, img_idx)

    visualizer.set_dataset_by_name(ds_name)
    visualizer.set_image_by_index(img_idx)
    visualizer.show()
    
fig = plt.figure(figsize=(18, 10))
suptitle = fig.suptitle(f"[{NAME}] Scatter plot for dice score by polyp size", fontsize="x-large")
for i, (ds_name, value) in enumerate(result_combined.items()):
    ax = fig.add_subplot(2,3,i+1)
    ax.scatter(value["size"], value["dice"], picker=True)
    ax.set_title(ds_name)
    ax.set_xlabel("Polyp size")
    ax.set_ylabel("Dice score")

fig.canvas.mpl_connect('pick_event', onpick)
plt.show()


