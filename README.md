# Polyp Segmentation using Mutiple resolution strategy - HRSeg
In this project, we propose a novel method called HRSeg for polyp segmentation that uses multiple-resolution images to improve the accuracy of polyp segmentation. The proposed method outperforms the state-of-the-art methods for polyp segmentation, achieving higher about 3% accuracy.

## Training strategy
![HRSeg training strategy](images/HRSeg-train.drawio.png)

## Inference strategy
![HRSeg inference strategy](images/HRSeg-infer.drawio.png)

## Pretrained model weight
Our HRSeg model weight can be download here [GG drive](https://drive.google.com/file/d/1L3C_Tnl0UcePhueXSkQlMykK7iv73eUB/view?usp=sharing)


## How to

Train

```sh
# Check arguments with
python train.py --help
```

Inference

```sh
python inference.py -n HRSeg -p path/to/model.pth
```

Analysis

```sh
python analysis.py -n1 HRSeg -n2 ssformer_S `
--print_table `
--show_scatter_dice_by_size `
--show_delta_dice `
--show_scatter_dice_by_range_size
```

Visualize

```sh
python visualize.py -n HRSeg -p path/to/model.pth
```