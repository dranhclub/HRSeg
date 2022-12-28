import cv2
import os
import numpy as np
from utils import dice
from utils import get_test_raw_dataset, get_train_raw_dataset, RawResult
from utils import DS_NAMES, CAPTURE_ROOT

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

class Visualizer():
    """
    Visualize image, grouth truth and prediction

    How to use:
    [a, d]: change image index
    [q, e]: change dataset index
    [+, -]: change image scale
    [g]   : show ground truth
    [p]   : show predict
    [ESQ] : quit
    """

    def __init__(self, name):
        self.name = name
        self.scale = 1
        self.show_gt = True
        self.show_pred = True
        self.winname = f"Visualization for {name}"
        self.capture_dir = os.path.join(CAPTURE_ROOT, name)
        self.text_color = (189, 255, 206)
        self.test_raw_dataset = get_test_raw_dataset()
        self.raw_result = RawResult(name)
        self.set_dataset_by_index(0)
        
        self.frame = self.render_frame()

    def set_dataset_by_name(self, ds_name):
        index = DS_NAMES.index(ds_name)
        self.set_dataset_by_index(index)

    def set_dataset_by_index(self, index):
        self.ds_idx = index % len(DS_NAMES)
        self.ds_name = DS_NAMES[self.ds_idx]
        self.n_imgs = len(self.test_raw_dataset.filenames[self.ds_name])
        self.set_image_by_index(0)

    def set_image_by_index(self, index):
        self.img_idx = index % self.n_imgs

    def capture(self):
        # Make dir
        os.makedirs(self.capture_dir, exist_ok=True)
        # Save
        save_path = os.path.join(self.capture_dir, f"{self.ds_name}.{self.filename}")
        cv2.imwrite(save_path, self.frame)
        print("Saved", save_path)

    def render_frame(self):
        # Get filename
        self.filename = self.test_raw_dataset.filenames[self.ds_name][self.img_idx]

        # Read imgs
        img = self.test_raw_dataset.get_img(self.ds_name, self.img_idx)
        gt = self.test_raw_dataset.get_gt(self.ds_name, self.img_idx)
        pred = self.raw_result.get_result(self.ds_name, self.img_idx)

        # Get original weight, height
        img_height, img_width = img.shape[0], img.shape[1]

        # Resize
        img = cv2.resize(img, None, fx=self.scale, fy=self.scale)
        gt = cv2.resize(gt, None, fx=self.scale, fy=self.scale)
        pred = cv2.resize(pred, None, fx=self.scale, fy=self.scale)
        
        # Get resized image width, height
        img_scaled_height, img_scaled_width = img.shape[0], img.shape[1]

        # Frame to show
        frame = np.copy(img)

        # Blend gt
        if self.show_gt:
            frame = blend(frame, (255, 255, 255), gt, alpha=0.85)
        
        # Blend prediction
        if self.show_pred:
            frame = blend(frame, (0, 255, 0), pred, alpha=0.5)

        # Text: dataset, image, scale, size
        text1 = f"Dataset {self.ds_idx}: {self.ds_name}"
        text2 = f"Image {self.img_idx}/{self.n_imgs - 1}: {self.filename}"
        text3 = f"Scale {self.scale}x (origin size: {img_width}x{img_height})"
        cv2.putText(img=frame, text=text1, org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)
        cv2.putText(img=frame, text=text2, org=(10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)
        cv2.putText(img=frame, text=text3, org=(10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)

        # Text: percent of white pixel
        percent = gt[gt > 0].size / gt.size * 100
        text = f"Size: {percent:.1f} %"
        cv2.putText(img=frame, text=text, org=(10, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)

        # Text: Dice score
        dice_score = dice(pred / 255, gt / 255) * 100
        text = f"Dice score: {dice_score:.2f} %"
        cv2.putText(img=frame, text=text, org=(10, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)

        # Text: Experiment name
        text = f"Experiment name: {self.name}"
        cv2.putText(img=frame, text=text, org=(10, img_scaled_height - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=self.text_color)

        return frame

    def show(self):
        while True:
            # Show
            frame = self.render_frame()
            cv2.imshow(self.winname, frame)

            # Handle event
            key = cv2.waitKey(0) & 0xFF
            if key == ord("d"):
                self.set_image_by_index(self.img_idx + 1)
            elif key == ord("a"):
                self.set_image_by_index(self.img_idx - 1)
            elif key == ord("q"):
                self.set_image_by_index(0)
                self.set_dataset_by_index(self.ds_idx - 1)
            elif key == ord("e"):
                self.set_image_by_index(0)
                self.set_dataset_by_index(self.ds_idx + 1)
            elif key == ord("+"):
                self.scale += 0.25
            elif key == ord("-"):
                self.scale -= 0.25
            elif key == ord("g"):
                self.show_gt = not self.show_gt
            elif key == ord("p"):
                self.show_pred = not self.show_pred
            elif key == ord("c"): # Capture 
                self.capture()
            elif key == 27:
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    v = Visualizer("e_40.Dec21-15h15.cutmix.sunseg")
    v.show()