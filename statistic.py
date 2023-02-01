# Statistic and analysis results

import cv2
import matplotlib.pyplot as plt
from utils import dice, get_test_raw_dataset, RawResult
from visualize import Visualizer
from utils import DS_NAMES
import argparse
import numpy as np
import pickle
import os

class Record():
    def __init__(self, name, ds_name, img_idx, dice, size):
        self.name = name
        self.ds_name = ds_name
        self.img_idx = img_idx
        self.dice = dice
        self.size = size


class Statistic():
    def __init__(self, name) -> None:
        self.name = name
        # self.test_raw_dataset = get_test_raw_dataset()
        self.raw_result = RawResult(name)

    @staticmethod
    def dice_vs_size(name):
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
            raw_result = RawResult(name)

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

    @staticmethod
    def _calc(name):
        print("Calc ", name)

        CACHE_DIR = './cache'
        cache_file = os.path.join(CACHE_DIR, name)

        if os.path.exists(cache_file):
            with open(cache_file, mode='rb') as f:
                records = pickle.load(f)
            print("Load from cache")
        else:
            records = []

            # For each dataset
            for ds_name in DS_NAMES:
                print("Processing", ds_name)

                test_raw_dataset = get_test_raw_dataset()
                raw_result = RawResult(name)

                # For each img
                for img_idx in range(len(test_raw_dataset.filenames[ds_name])):

                    # Read GT, pred
                    gt = test_raw_dataset.get_gt(ds_name, img_idx)
                    pred = raw_result.get_result(ds_name, img_idx)

                    # Percentage of white pixel per total number of pixel
                    percent = gt[gt > 0].size / gt.size * 100
                    
                    # Dice score
                    dice_score = dice(gt / 255, pred / 255) * 100

                    # Record
                    records.append(Record(name, ds_name, img_idx, dice_score, percent))

            print("Process done!")

            # Save to cache
            with open(cache_file, mode='wb') as f:
                pickle.dump(records, f)

        return records

    @staticmethod
    def calc_test_polyp_size():
        result = {
            'CVC-300': [], 
            'CVC-ClinicDB': [], 
            'Kvasir': [], 
            'CVC-ColonDB': [], 
            'ETIS-LaribPolypDB': [],
        }

        # For each dataset
        for ds_name in DS_NAMES:
            print("Calc size", ds_name)

            test_raw_dataset = get_test_raw_dataset()

            # For each img
            for img_idx in range(len(test_raw_dataset.filenames[ds_name])):

                # Read GT
                gt = test_raw_dataset.get_gt(ds_name, img_idx)

                # Percentage of white pixel per total number of pixel
                percent = gt[gt > 0].size / gt.size * 100
                result[ds_name].append(percent)

        print("Process done!")
        return result

    @staticmethod
    def calc_dice(name):
        result = {
            'CVC-300': [], 
            'CVC-ClinicDB': [], 
            'Kvasir': [], 
            'CVC-ColonDB': [], 
            'ETIS-LaribPolypDB': [],
        }

        # For each dataset
        for ds_name in DS_NAMES:
            print("Calc dice", ds_name)

            test_raw_dataset = get_test_raw_dataset()
            raw_result = RawResult(name)

            # For each img
            for img_idx in range(len(test_raw_dataset.filenames[ds_name])):

                # Read GT, pred
                gt = test_raw_dataset.get_gt(ds_name, img_idx)
                pred = raw_result.get_result(ds_name, img_idx)

                # Dice score
                dice_score = dice(gt / 255, pred / 255) * 100
                result[ds_name].append(dice_score)

        print("Process done!")
        return result

    def combine(self, result):
        new_result = result.copy()
        new_result["all"] = {
            "size": [],
            "dice": []
        }
        for ds_name, value in result.items():
            new_result["all"]["size"].extend(value["size"])
            new_result["all"]["dice"].extend(value["dice"])
        return new_result

    def combine_dice(self, result):
        new_result = result.copy()
        new_result["all"] = []
        for ds_name, value in result.items():
            new_result["all"].extend(value)
        return new_result

    def show_scatter_dice_by_size(self, name_to_compare):
        result = self.combine(Statistic.dice_vs_size(self.name))
        visualizer0 = Visualizer(NAME)
        visualizer1 = Visualizer(NAME)
        visualizer2 = Visualizer(name_to_compare)

        def onpick(event):
            hit_list = event.ind
            img_idx = hit_list[0]
            ds_name = event.artist.axes.get_title()
            print(ds_name, img_idx)

            visualizer0.set_dataset_by_name(ds_name)
            visualizer0.set_image_by_index(img_idx)
            visualizer1.set_dataset_by_name(ds_name)
            visualizer1.set_image_by_index(img_idx)
            visualizer1.show_gt = False
            visualizer1.show_pred = False
            visualizer2.set_dataset_by_name(ds_name)
            visualizer2.set_image_by_index(img_idx)

            # Show
            frame1 = visualizer1.render_frame()
            if frame1.shape[1] > 600:
                visualizer0.scale = 0.5
                visualizer1.scale = 0.5
                visualizer2.scale = 0.5
            else:
                visualizer0.scale = 1
                visualizer1.scale = 1
                visualizer2.scale = 1

            frame0 = visualizer0.render_frame()
            frame1 = visualizer1.render_frame()
            frame2 = visualizer2.render_frame()
            frame = np.hstack((frame0, frame1, frame2))

            cv2.imshow(f"{self.name} vs {name_to_compare}", frame)
            
        fig = plt.figure(figsize=(18, 10))
        suptitle = fig.suptitle(f"[{self.name}] Scatter plot for dice score by polyp size", fontsize="x-large")
        for i, (ds_name, value) in enumerate(result.items()):
            ax = fig.add_subplot(2,3,i+1)
            ax.scatter(value["size"], value["dice"], picker=True)
            ax.set_title(ds_name)
            ax.set_xlabel("Polyp size")
            ax.set_ylabel("Dice score")

        fig.canvas.mpl_connect('pick_event', onpick)
        plt.show()

    def show_curve_delta_dice(self, name_to_compare):
        records_1 = Statistic._calc(NAME)
        records_2 = Statistic._calc(name_to_compare)

        visualizer0 = Visualizer(NAME)
        visualizer1 = Visualizer(NAME)
        visualizer2 = Visualizer(name_to_compare)

        delta_dices = {}

        def onpick(event):
            hit_list = event.ind
            hit_idx = hit_list[0]
            ds_name = event.artist.axes.get_title()
            r1 = delta_dices[ds_name][hit_idx]["r1"]
            img_idx = r1.img_idx
            print(ds_name, img_idx)

            visualizer0.set_dataset_by_name(ds_name)
            visualizer0.set_image_by_index(img_idx)
            visualizer1.set_dataset_by_name(ds_name)
            visualizer1.set_image_by_index(img_idx)
            visualizer1.show_gt = False
            visualizer1.show_pred = False
            visualizer2.set_dataset_by_name(ds_name)
            visualizer2.set_image_by_index(img_idx)

            # Show
            frame1 = visualizer1.render_frame()
            if frame1.shape[1] > 600:
                visualizer0.scale = 0.5
                visualizer1.scale = 0.5
                visualizer2.scale = 0.5
            else:
                visualizer0.scale = 1
                visualizer1.scale = 1
                visualizer2.scale = 1

            frame0 = visualizer0.render_frame()
            frame1 = visualizer1.render_frame()
            frame2 = visualizer2.render_frame()
            frame = np.hstack((frame0, frame1, frame2))

            cv2.imshow(f"{self.name} vs {name_to_compare}", frame)

        fig = plt.figure(figsize=(18, 10))
        suptitle = fig.suptitle(f"Delta dice [{name_to_compare} - {self.name}]", fontsize="x-large")
        

        for i, ds_name in enumerate(DS_NAMES):
            list_record_by_ds_1 = list(filter(lambda r: r.ds_name == ds_name, records_1))
            list_record_by_ds_2 = list(filter(lambda r: r.ds_name == ds_name, records_2))

            delta_dice = []
            for r1, r2 in zip(list_record_by_ds_1, list_record_by_ds_2):
                delta_dice.append({
                    "delta_dice": r2.dice - r1.dice,
                    "size": r1.size,
                    "r1": r1,
                    "r2": r2
                })
            
            # sort by size
            delta_dice.sort(key=lambda x: x['size'])
            # Save
            delta_dices[ds_name] = delta_dice
            
        for i, ds_name in enumerate(DS_NAMES):
            delta_dice = delta_dices[ds_name]
            N = len(delta_dice)
            ax = fig.add_subplot(2,3,i+1)
            ax.scatter(np.arange(0, N, 1), list(map(lambda x: x['delta_dice'], delta_dice)), picker=True)
            ax.set_title(ds_name)
            # ax.set_xlabel("Image")
            ax.set_ylabel("Delta dice")
            ax.axhline(y=0, color='r', linestyle='-')

        fig.canvas.mpl_connect('pick_event', onpick) 
        plt.show()

    def mean_dice_by_range_size(self, name):
        dbs_result = self.combine(self.dice_vs_size(name))
        dbrs_result = {
            'CVC-300': {}, 
            'CVC-ClinicDB': {}, 
            'Kvasir': {}, 
            'CVC-ColonDB': {}, 
            'ETIS-LaribPolypDB': {},
            'all': {}
        }
        size_milestones = np.arange(0, 100, 5)
        for i in range(len(size_milestones) - 1):
            s1 = size_milestones[i]
            s2 = size_milestones[i+1]
            range_label = f"{s1}-{s2}"
            for ds_name, value in dbs_result.items():
                dices = []
                for j, s in enumerate(value['size']):
                    if s1 < s <= s2:
                        dices.append(value['dice'][j])
                dbrs_result[ds_name][range_label] = np.mean(dices)
        return dbrs_result

    def show_scattter_dice_by_range_size(self, name_to_compare):
        dbrs_result_1 = self.mean_dice_by_range_size(self.name)
        dbrs_result_2 = self.mean_dice_by_range_size(name_to_compare)

        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(f"Mean dice by range size [{self.name} (blue) and {name_to_compare} (red)]", fontsize="x-large")
        for i, (ds_name, value1) in enumerate(dbrs_result_1.items()):
            value2 = dbrs_result_2[ds_name]
            
            ax = fig.add_subplot(2,3,i+1)
                        
            for label, m in value1.items():
                ax.scatter(label, m, color='b')
            
            for label, m in value2.items():
                ax.scatter(label, m, color='r')

            ax.set_title(ds_name)
            ax.set_ylabel("Mean dice")
            ax.xaxis.set_tick_params(rotation=45)
        plt.show()
    
    def print_dice(self):
        result_1 = self.combine_dice(self.calc_dice(self.name))
        print("============Mean dice============")
        for ds_name, value in result_1.items():
            print(f"{ds_name}: {np.mean(value):.2f}%")
        print("=================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Statistic")
    parser.add_argument("--name", "-n", type=str, default="HRSeg6")
    parser.add_argument("--compare_to", "-c", type=str, default="e_120.Dec20.albumen")
    opt = parser.parse_args()

    NAME = opt.name
    NAME2 = opt.compare_to
    print("NAME =", NAME)
    print("Compare to", NAME2)

    stat = Statistic(NAME)
    stat.print_dice()
    stat.show_scatter_dice_by_size(NAME2)
    # stat.show_curve_delta_dice(NAME2)
    # stat.show_scattter_dice_by_range_size(NAME2)