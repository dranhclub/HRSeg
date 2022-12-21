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
    def __init__(self, dataset_roots, train_size):
        self.train_size = train_size

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
        self.tf0 = A.RandomScale(p=0.5, interpolation=0, scale_limit=(-0.5, -0.5))
        self.tf1 = A.Compose([
            A.PadIfNeeded(p=1.0, min_height=train_size, min_width=train_size),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Rotate(90),
            A.Spatter(p=0.1),
            A.Resize(self.train_size, self.train_size),
        ])
        self.tf2 = A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.tf3 = albumentations.pytorch.transforms.ToTensorV2(transpose_mask=True)


    def __getitem__(self, index):
        # Read image and mask
        image = cv2.imread(self.images[index], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(self.gts[index], cv2.IMREAD_GRAYSCALE)
        gt = np.expand_dims(gt, 2)
        gt = (gt / 255).astype("float32")
        
        # Apply downscale if polyp size > 2%
        gt_np = np.array(gt)
        percent = gt_np[gt_np > 0].size / gt_np.size * 100
        if percent > 2:
            res_tf0 = self.tf0(image=image, mask=gt)
            image, gt = res_tf0['image'], res_tf0['mask']

        # Apply augment
        res_tf1 = self.tf1(image=image, mask=gt)
        image, gt = res_tf1['image'], res_tf1['mask']
        
        # Apply normalize by ImageNet's mean and std
        image = self.tf2(image=image)['image']

        # Apply to tensor
        res_tf3  = self.tf3(image=image, mask=gt)
        image, gt = res_tf3['image'], res_tf3['mask']

        return image, gt

    def __len__(self):
        return self.size


def get_train_loader(train_root, batchsize, train_size, shuffle=True, num_workers=4, pin_memory=True):

    dataset = TrainDataset(train_root, train_size)
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
