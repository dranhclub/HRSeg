# Visualize image, grouth truth and prediction
# How to use:
# [a, d]: change image index
# [q, e]: change dataset index
# [+, -]: change image scale
# [g]   : show ground truth
# [p]   : show predict
# [ESQ] : quit

import cv2
import os
import numpy as np
from utils import dice
from natsort import natsorted

def blend(background, foreground_color, mask, alpha):
    '''
    background: uint8 with shape (w x h x 3)
    foreground: example : (200, 0, 0)
    mask: uint8 with shape (w x h)
    alpha: transparent level of foreground_color in range 0-1
    
    return: uint8 blended image
    '''
    foreground = np.zeros_like(background)
    mask = np.repeat(np.expand_dims(mask, axis=2), repeats=3, axis=2)
    np.place(foreground, mask > 0, foreground_color)
    background = background / 255
    mask = mask / 255 * alpha
    foreground_color = np.array(foreground_color) / 255
    img = background * (1-mask) + foreground_color * mask
    img = np.uint8(img * 255)
    return img

DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

DS_PATH = './dataset/TestDataset'

PRED_PATH = './result_map/onestep' # result when inference 1 step
# PRED_PATH = './result_map/twostep' # result when inferenec 2 step
# PRED_PATH = './result_map/affine_big_only'

WINNAME = 'Baseline'
SAVE_PATH = 'captured/onestep'

ds_idx = 0
img_idx = 0
scale = 1
show_gt = True
show_pred = True

while True:
    ds_idx = ds_idx % len(DS_NAMES)

    # Get file name by index
    img_dir = f"{DS_PATH}/{DS_NAMES[ds_idx]}/images"
    imgs = natsorted(os.listdir(img_dir))
    img_idx = img_idx % len(imgs)
    img_filename = imgs[img_idx]
    img_path = f"{DS_PATH}/{DS_NAMES[ds_idx]}/images/{img_filename}"

    gt_dir = f"{DS_PATH}/{DS_NAMES[ds_idx]}/masks"
    gt_filename = imgs[img_idx]
    gt_path = f"{DS_PATH}/{DS_NAMES[ds_idx]}/masks/{gt_filename}"

    pred_dir = f"{PRED_PATH}/{DS_NAMES[ds_idx]}"
    pred_filename = imgs[img_idx]
    pred_path = f"{PRED_PATH}/{DS_NAMES[ds_idx]}/{pred_filename}"

    print(gt_path, pred_path)

    # Read imgs
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

    # Get original weight, height
    img_height, img_width = img.shape[0], img.shape[1]

    # Scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    gt = cv2.resize(gt, None, fx=scale, fy=scale)
    pred = cv2.resize(pred, None, fx=scale, fy=scale)
    
    # Get scaled width, height
    img_scaled_height, img_scaled_width = img.shape[0], img.shape[1]

    # Frame to show
    frame = np.copy(img)

    # Blend gt
    if show_gt:
        frame = blend(frame, (255, 255, 255), gt, alpha=0.85)
    
    # Blend prediction
    if show_pred:
        frame = blend(frame, (0, 255, 0), pred, alpha=0.5)

    # Text: dataset, image, scale
    text1 = f"Dataset {ds_idx}: {DS_NAMES[ds_idx]}"
    text2 = f"Image {img_idx}/{len(imgs) - 1}: {gt_filename}"
    text3 = f"Scale {scale}x (origin size: {img_width}x{img_height})"
    cv2.putText(img=frame, text=text1, org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=frame, text=text2, org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=frame, text=text3, org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))

    # Text: percent of white pixel
    percent = gt[gt > 0].size / gt.size * 100
    text = f"Size: {percent:.1f} %"
    cv2.putText(img=frame, text=text, org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))

    # Text: Dice score
    dice_score = dice(pred / 255, gt / 255) * 100
    text = f"Dice score: {dice_score:.2f} %"
    cv2.putText(img=frame, text=text, org=(10, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))

    # Text: Prediction folder
    text = f"Pred: {PRED_PATH}"
    cv2.putText(img=frame, text=text, org=(10, img_scaled_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))

    # Show
    cv2.imshow(WINNAME, frame)

    key = cv2.waitKey(0) & 0xFF
    if key == ord("d"):
        img_idx += 1
    elif key == ord("a"):
        img_idx -= 1
    elif key == ord("q"):
        img_idx = 0
        ds_idx -= 1
    elif key == ord("e"):
        img_idx = 0
        ds_idx += 1
    elif key == ord("+"):
        scale += 0.25
    elif key == ord("-"):
        scale -= 0.25
    elif key == ord("g"):
        show_gt = not show_gt
    elif key == ord("p"):
        show_pred = not show_pred
    elif key == ord("c"): # Capture 
        # Make dir
        os.makedirs(SAVE_PATH, exist_ok=True)
        # Save
        ds_name = DS_NAMES[ds_idx]
        path_to_save = os.path.join(SAVE_PATH, f"{ds_name}.{img_filename}")
        cv2.imwrite(path_to_save, frame)
        print("Saved", path_to_save)
    elif key == 27:
        break


cv2.destroyAllWindows()