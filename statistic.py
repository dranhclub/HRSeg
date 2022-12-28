# Statistic and analysis results

import cv2
import matplotlib.pyplot as plt
from utils import dice, get_test_raw_dataset, get_train_raw_dataset, RawResult
from visualize import Visualizer
from utils import DS_NAMES
import argparse
import numpy as np

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

        test_raw_dataset = get_test_raw_dataset()
        raw_result = RawResult(NAME)

        # For each img
        for img_idx in range(len(test_raw_dataset.filenames[ds_name])):

            # Read GT, pred
            gt = test_raw_dataset.get_gt(ds_name, img_idx)
            pred = raw_result.get_result(ds_name, img_idx)

            # Percentage of white pixel per total number of pixel
            percent = gt[gt > 0].size / gt.size * 100
            result[ds_name]['size'].append(percent)
            
            # Dice score
            dice_score = dice(gt / 255, pred / 255) * 100
            result[ds_name]['dice'].append(dice_score)

    print("Process done!")
    return result

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

def calc_mean_and_print(result):
    print("=====Mean dice=====")
    for ds_name, value in result.items():
        m = np.mean(value["dice"])
        print(f"{ds_name}: {m:.2f} %")   
    print("===================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Statistic")
    parser.add_argument("--name", type=str, default="PolypPVT")
    opt = parser.parse_args()

    NAME = opt.name
    print("NAME =", NAME)
    
    ########## Calculate
    result = dice_vs_size()
    result_combined = combine(result)
    calc_mean_and_print(result_combined)

    ########## Plot and visualize
    visualizer = Visualizer(NAME)

    def onpick(event):
        hit_list = event.ind
        img_idx = hit_list[0]
        ds_name = event.artist.axes.get_title()
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


