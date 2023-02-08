# High Resolution Polyp Segmentation

## How to

Train

```sh
python train.py
```

Inference

```sh
python inference.py -n HRSeg9 -p model_pth/HRSeg9.e_40.Feb07-03h53.pth
python inference.py -n HRSeg10 -p model_pth/HRSeg10.e_40.Feb08-05h23.pth
```

Analysis

```sh
python analysis.py -n1 PolypPVT -n2 HRSeg10 `
--print_table `
--show_scatter_dice_by_size `
--show_delta_dice `
--show_scatter_dice_by_range_size

python -m debugpy --listen 5678 --wait-for-client analysis.py -n1 HRSeg8 -n2 ssformer_S --print_table
```

Visualize

```sh
python visualize.py -n HRSeg10 -p model_pth/HRSeg10.e_40.Feb08-05h23.pth
```
