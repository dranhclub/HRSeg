import os
import torch.utils.data as data
import numpy as np
from natsort import natsorted
import albumentations as A
import albumentations.pytorch.transforms
import cv2    

class TrainDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, dataset_roots, train_size, batch_size):
        self.train_size = train_size
        self.batch_size = batch_size
        self.seed = np.random.randint(0, 1000)
        self.counter = 0

        ##### Get file paths
        self.images = []
        self.gts = []

        for dataset_root in dataset_roots:
            image_root = os.path.join(dataset_root, "images")
            gt_root = os.path.join(dataset_root, "masks")
            
            for f in os.listdir(image_root):
                if f.endswith('.jpg') or f.endswith('.png'):
                    file = os.path.join(image_root, f)
                    self.images.append(file)

            for f in os.listdir(gt_root):
                if f.endswith('.png'):
                    file = os.path.join(gt_root, f)
                    self.gts.append(file)

        self.images = natsorted(self.images)
        self.gts = natsorted(self.gts)
        self.size = len(self.images)

        ###### Transforms
        self.tf_augment = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(90, border_mode=None),
            A.PadIfNeeded(min_height=576, min_width=576, border_mode=None),
            A.RandomCrop(576, 576),
        ])
        self.tf_norm = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tf_resize = A.Resize(train_size, train_size, cv2.INTER_CUBIC)
        self.tf_to_tensor = albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)

    def __getitem__(self, index):
        # Read image and mask
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        gt = np.expand_dims(gt, 2)
        gt = (gt / 255).astype("float32")

        # Apply normalize by ImageNet's mean and std
        image = self.tf_norm(image=image)['image']

        # Apply augment
        res_tf_augment = self.tf_augment(image=image, mask=gt)
        image, gt = res_tf_augment['image'], res_tf_augment['mask']

        # Random crop inner image to 288
        inner_image, x0, y0, x1, y1 = self.memorize_crop(image)

        # Resize outer image to 288
        outer_image = self.tf_resize(image=image)['image']

        # Apply to tensor
        gt = self.tf_to_tensor(image=gt)['image']

        inner_image = self.tf_to_tensor(image=inner_image)['image']
        outer_image = self.tf_to_tensor(image=outer_image)['image']
        
        return {
            "image": outer_image, 
            "inner_image": inner_image, 
            "mask": gt,
            "slice": np.array([x0, y0, x1, y1])
        }

    def memorize_crop(self, image):
        np.random.seed(self.seed + self.counter // self.batch_size)
        self.counter += 1
        x0, y0 = np.random.randint(0, 448-352, size=2, dtype=np.uint8)
        x1 = x0 + self.train_size
        y1 = y0 + self.train_size
        inner_image = image[y0:y1, x0:x1]
        return inner_image, x0, y0, x1, y1

    def __len__(self):
        return self.size


def get_train_loader(train_roots, batchsize, train_size, shuffle=True, num_workers=4, pin_memory=True):

    dataset = TrainDataset(train_roots, train_size, batchsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

class TestDatasets():
    def __init__(self, test_root, train_size):
        self.DS_NAMES = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
        self.test_root = test_root
        self.train_size = train_size
        
        # Crete datasets object
        datasets = {}
        for name in self.DS_NAMES:
            root = os.path.join(test_root, name)
            imgs = os.listdir(os.path.join(root, "images"))
            n_imgs = len(imgs)
            datasets[name] = {
                "root": root,
                "imgs": imgs,
                "n_imgs": n_imgs
            }
        self.datasets = datasets

        # Transform 
        self.img_transform = A.Compose([
            A.Resize(self.train_size, self.train_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            albumentations.pytorch.transforms.ToTensorV2(),
        ])

    def get_item(self, ds_name, index):
        dataset = self.datasets[ds_name]
        root = dataset["root"]
        imgs = dataset["imgs"]
        img_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "masks")

        img_path = os.path.join(img_dir, imgs[index])
        gt_path = os.path.join(gt_dir, imgs[index])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt / 255).astype("float32")

        transformed_image = self.img_transform(image=image)["image"]
        
        return image, transformed_image, gt

    def get_item_by_name(self, ds_name, img_name):
        dataset = self.datasets[ds_name]
        root = dataset["root"]
        img_dir = os.path.join(root, "images")
        gt_dir = os.path.join(root, "masks")

        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt = (gt / 255).astype("float32")

        transformed_image = self.img_transform(image=image)["image"]
        
        return image, transformed_image, gt
