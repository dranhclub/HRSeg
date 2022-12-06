# Visualize image, grouth truth and prediction
# How to use:
# [a, d]: change image index
# [q, e]: change dataset index
# [+, -]: change image scale
# ESQ: quit

import cv2
import os
import numpy as np

DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']

GT_PATH = './dataset/TestDataset'
PRED_PATH = './result_map/PolypPVT'

ds_idx = 0
img_idx = 0
scale = 1
show_gt = True

cv2.namedWindow("Image and GT")
cv2.namedWindow("Predict")

while True:
    ds_idx = ds_idx % len(DS_NAMES)

    # Get file name by index
    img_dir = f"{GT_PATH}/{DS_NAMES[ds_idx]}/images"
    imgs = os.listdir(img_dir)
    img_idx = img_idx % len(imgs)
    img_filename = imgs[img_idx]
    img_path = f"{GT_PATH}/{DS_NAMES[ds_idx]}/images/{img_filename}"

    gt_dir = f"{GT_PATH}/{DS_NAMES[ds_idx]}/masks"
    gt_filename = imgs[img_idx]
    gt_path = f"{GT_PATH}/{DS_NAMES[ds_idx]}/masks/{gt_filename}"

    pred_dir = f"{PRED_PATH}/{DS_NAMES[ds_idx]}"
    imgs = os.listdir(pred_dir)
    pred_filename = imgs[img_idx]
    pred_path = f"{PRED_PATH}/{DS_NAMES[ds_idx]}/{pred_filename}"

    print(gt_path, pred_path)

    # Read imgs
    img = cv2.imread(img_path)
    gt = cv2.imread(gt_path)
    pred = cv2.imread(pred_path)

    # Scale
    img = cv2.resize(img, None, fx=scale, fy=scale)
    gt = cv2.resize(gt, None, fx=scale, fy=scale)
    pred = cv2.resize(pred, None, fx=scale, fy=scale)
    
    # Blend img + gt
    if show_gt:
        mask = gt
        mask[mask > 0] = 255
        np.place(mask, mask == (255,255,255), (200,0,0))
        img = img / 255
        mask = mask / 255
        img = img * (1-mask) + mask * mask
        img = np.uint8(img * 255)

    # Draw text
    text1 = f"Dataset {ds_idx}: {DS_NAMES[ds_idx]}"
    text2 = f"Image {img_idx}/{len(imgs) - 1}: {gt_filename}"
    text3 = f"{scale}x"
    cv2.putText(img=img, text=text1, org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=img, text=text2, org=(40, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=img, text=text3, org=(40, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=pred, text=text1, org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=pred, text=text2, org=(40, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))
    cv2.putText(img=pred, text=text3, org=(40, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255))

    # Show
    cv2.imshow("Image and GT", img)
    cv2.imshow("Predict", pred)

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
    elif key == ord(" "):
        show_gt = not show_gt
    elif key == 27:
        break


cv2.destroyAllWindows()