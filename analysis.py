import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prettytable import PrettyTable

from utils import DS_NAMES, RawResult, dice, get_test_raw_dataset
from visualize import Visualizer


def calc(name):
    """
    Calculate dice score and size of polyp
    
    params:
        name: experiment name
    """
    print("Processing ", name)
    CACHE_DIR = './cache'
    cache_file = os.path.join(CACHE_DIR, name + ".csv")

    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, index_col=0)
        print(f"Loaded calc result of {name} from cache")
    else:
        rows = []
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
                size = gt[gt > 0].size / gt.size * 100
                
                # Dice score
                dice_score = dice(gt / 255, pred / 255) * 100

                # Record
                rows.append((name, ds_name, img_idx, dice_score, size))

        # Create dataframe
        df = pd.DataFrame(rows, columns=["name", "ds_name", "img_idx", "dice", "size"])
        df.to_csv(cache_file)
        print("Process done!")

    return df

class Comparator():
    def __init__(self, name1, name2):
        self.name1 = name1
        self.name2 = name2
        self.df = pd.concat([calc(name1), calc(name2)])

        self.visualizer0 = Visualizer(self.name1)
        self.visualizer1 = Visualizer(self.name1)
        self.visualizer2 = Visualizer(self.name2)
        
        self.visualizer1.show_gt = False
        self.visualizer1.show_pred = False

    def print_dice_table(self):
        table = PrettyTable()
        table.field_names = ["Name", *DS_NAMES, "Average"]

        dicelist1 = []
        df2 = self.df[self.df['name']==self.name1].groupby(['ds_name']).mean().drop(columns=['img_idx', 'size'])
        for ds_name in DS_NAMES:
            dicelist1.append(f"{df2['dice'][ds_name]:.2f}")
        m = self.df[self.df['name']==self.name1]['dice'].mean()
        dicelist1.append(f"{m:.2f}")

        dicelist2 = []
        df2 = self.df[self.df['name']==self.name2].groupby(['ds_name']).mean().drop(columns=['img_idx', 'size'])
        for ds_name in DS_NAMES:
            dicelist2.append(f"{df2['dice'][ds_name]:.2f}")
        m = self.df[self.df['name']==self.name2]['dice'].mean()
        dicelist2.append(f"{m:.2f}")

        table.add_row([self.name1] + dicelist1)
        table.add_row([self.name2] + dicelist2)
        
        print("Dice table")
        print(table)

    def _show_visualize(self, ds_name, img_idx):
        self.visualizer0.set_dataset_by_name(ds_name)
        self.visualizer0.set_image_by_index(img_idx)
        self.visualizer1.set_dataset_by_name(ds_name)
        self.visualizer1.set_image_by_index(img_idx)
        self.visualizer1.show_gt = False
        self.visualizer1.show_pred = False
        self.visualizer2.set_dataset_by_name(ds_name)
        self.visualizer2.set_image_by_index(img_idx)

        # Show
        frame1 = self.visualizer1.render_frame()
        if frame1.shape[1] > 600:
            self.visualizer0.scale = 0.5
            self.visualizer1.scale = 0.5
            self.visualizer2.scale = 0.5
        else:
            self.visualizer0.scale = 1
            self.visualizer1.scale = 1
            self.visualizer2.scale = 1

        frame0 = self.visualizer0.render_frame()
        frame1 = self.visualizer1.render_frame()
        frame2 = self.visualizer2.render_frame()
        frame = np.hstack((frame0, frame1, frame2))

        winname = f"{self.name1} vs {self.name2}"
        cv2.imshow(winname, frame)
        cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)

    def show_scatter_dice_by_size(self):
        name = self.name1

        def onpick(event):
            hit_list = event.ind
            img_idx = hit_list[0]
            ds_name = event.artist.axes.get_title()
            self._show_visualize(ds_name, img_idx)
            
        fig, axes = plt.subplots(2, 3)
        fig.set_size_inches(18, 10)
        fig.suptitle(f"[{name}] Scatter plot for dice score by polyp size", fontsize="x-large")
        for i, ds_name in enumerate(DS_NAMES + ['All']):
            df = self.df
            if i != 5:
                records = df.loc[(df['name']==name) & (df['ds_name']==ds_name)]
            else:
                records = df.loc[(df['name']==name)]
            sizelist = records['size']
            dicelist = records['dice']
            ax = axes[i // 3][i % 3]
            ax.scatter(sizelist, dicelist, picker=True)
            ax.set_title(ds_name)
            ax.set_xlabel("Polyp size")
            ax.set_ylabel("Dice score")

        fig.canvas.mpl_connect('pick_event', onpick)
        fig.tight_layout()
        plt.show()

    def show_delta_dice(self):
        delta_dices = {}

        def onpick(event):
            hit_list = event.ind
            hit_idx = hit_list[0]
            ds_name = event.artist.axes.get_title()
            r1 = delta_dices[ds_name][hit_idx]["r1"]
            img_idx = r1.img_idx
            self._show_visualize(ds_name, img_idx)

        fig, axes = plt.subplots(2, 3)
        fig.set_size_inches(18, 10)
        fig.suptitle(f"Delta dice [{self.name1} - {self.name2}]", fontsize="x-large")
        
        df = self.df
        for i, ds_name in enumerate(DS_NAMES + ["All"]):
            if i != 5:
                df1 = df.loc[(df['name']==self.name1) & (df['ds_name']==ds_name)]
                df2 = df.loc[(df['name']==self.name2) & (df['ds_name']==ds_name)]
            else:
                df1 = df.loc[(df['name']==self.name1)]
                df2 = df.loc[(df['name']==self.name2)]

            delta_dice = []
            for i in range(len(df1)):
                r1 = df1.iloc[i]
                r2 = df2.iloc[i]

                delta_dice.append({
                    "delta_dice": r1['dice'] - r2['dice'],
                    "size": r1['size'],
                    "r1": r1,
                    "r2": r2
                })
            
            # sort by size
            delta_dice.sort(key=lambda x: x['size'])
            # Save
            delta_dices[ds_name] = delta_dice
            
        for i, ds_name in enumerate(DS_NAMES + ["All"]):
            delta_dice = delta_dices[ds_name]
            N = len(delta_dice)
            ax = axes[i // 3][i % 3]
            ax.scatter(np.arange(0, N, 1), list(map(lambda x: x['delta_dice'], delta_dice)), picker=True)
            ax.set_title(ds_name)
            ax.set_xlabel("Polyp size")
            ax.set_ylabel("Delta dice")
            ax.axhline(y=0, color='r', linestyle='-')

        fig.canvas.mpl_connect('pick_event', onpick) 
        fig.tight_layout()
        plt.show()

    def show_scatter_dice_by_range_size(self):
        fig, axes = plt.subplots(2, 3)
        fig.set_size_inches(18, 10)
        fig.suptitle(f"Mean dice by range size [{self.name1} vs {self.name2}]", fontsize="x-large")
        df = self.df
        for i, ds_name in enumerate(DS_NAMES + ['All']):
            ax = axes[i // 3][i % 3]
            for name in [self.name1, self.name2]:
                if i != 5:
                    df1 = df[(df['name'] == name) & (df['ds_name'] == ds_name)]
                else:
                    df1 = df[df['name'] == name]
                mean_dice_by_interval = df1.groupby(pd.cut(df1['size'], np.arange(0, 100, 5))).mean().dropna()['dice']
                labels = mean_dice_by_interval.index.astype(str).to_list()
                values = mean_dice_by_interval.to_list()
                ax.scatter(labels, values, label=name)
            
            ax.legend()
            ax.set_title(ds_name)
            ax.set_xlabel("Polyp size")
            ax.set_ylabel("Mean dice")
            ax.xaxis.set_tick_params(rotation=45)
        
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Statistic")
    parser.add_argument("--name1", "-n1", type=str, default="HRSeg8")
    parser.add_argument("--name2", "-n2", type=str, default="ssformer_S")
    parser.add_argument("--print_table", action='store_true')
    parser.add_argument("--show_scatter_dice_by_size", action='store_true')
    parser.add_argument("--show_delta_dice", action='store_true')
    parser.add_argument("--show_scatter_dice_by_range_size", action='store_true')

    opt = parser.parse_args()

    comparator = Comparator(opt.name1, opt.name2)

    if opt.print_table:
        comparator.print_dice_table()

    if opt.show_scatter_dice_by_size:
        comparator.show_scatter_dice_by_size()

    if opt.show_delta_dice:
        comparator.show_delta_dice()
    
    if opt.show_scatter_dice_by_range_size:
        comparator.show_scatter_dice_by_range_size()