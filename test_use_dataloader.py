from utils.dataloader import get_loader

image_root = './dataset/TrainDataset/images/'
gt_root = './dataset/TrainDataset/masks/'
batch_size = 8
train_size = 352
augmentation = 'True'
train_loader = get_loader(image_root, gt_root, batchsize=batch_size, trainsize=train_size,
                              augmentation=augmentation, num_workers=1)

images, targets = next(iter(train_loader))                        