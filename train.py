from utils import clip_gradient, adjust_lr
from dataloader import get_loader, test_dataset
from model import PolypSeg
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import argparse
from datetime import datetime
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def structure_loss(pred, mask):
    weit = 1 + 5 * \
        torch.abs(F.avg_pool2d(mask, kernel_size=31,
                  stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()


def test(model, test_ds_path):
    dice_record = {}
    for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        data_path = os.path.join(test_ds_path, dataset)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        model.eval()
        n_imgs = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        DSC = 0.0
        for i in range(n_imgs):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            res = model(image)
            # eval Dice
            res = F.upsample(res, size=gt.shape,
                             mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            input = res
            target = np.array(gt)
            smooth = 1
            input_flat = np.reshape(input, (-1))
            target_flat = np.reshape(target, (-1))
            intersection = (input_flat * target_flat)
            dice = (2 * intersection.sum() + smooth) / \
                (input.sum() + target.sum() + smooth)
            dice = '{:.4f}'.format(dice)
            dice = float(dice)
            DSC = DSC + dice

        dataset_dice = DSC / n_imgs

        print(dataset, ': ', dataset_dice)

        dice_record[dataset] = dataset_dice

    writer.add_scalars('Test dice', dice_record, global_step=epoch * n_steps_per_epoch)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for i, pack in enumerate(train_loader, start=1):

        # data prepare
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # forward
        out = model(images)
        loss = structure_loss(out, gts)

        # backward
        optimizer.zero_grad()
        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        # record loss
        loss_val = loss.data

        # log
        step = epoch * n_steps_per_epoch + i
        writer.add_scalar("Train loss", loss_val, step)
        if i % 20 == 0 or i == n_steps_per_epoch:
            print(
                f'{datetime.now()} Epoch [{epoch:03d}/{opt.epoch:03d}], Step [{i:04d}/{n_steps_per_epoch:04d}], loss: {loss_val:0.4f}]')


def save(model, save_path, epoch, start_timestamp):
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{model._get_name()}.e_{epoch}.{start_timestamp}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, file_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--n_epochs_per_test', type=float,
                        default=1, help='number of epochs per a test')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--save_path', type=str,
                        default='./model_pth',
                        help='path to save model weight')

    parser.add_argument('--num_workers', type=int,
                        default=1)
    
    opt = parser.parse_args()

    # Build model
    model = PolypSeg().cuda()

    # Optimizer
    params = model.parameters()
    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(
            params, opt.lr, weight_decay=1e-4, momentum=0.9)
    print(optimizer)

    # Dataloader
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)
    # train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    train_loader = get_loader(
        image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize, num_workers=1)

    # Start training
    print("#" * 20, "Start Training", "#" * 20)
    start_timestamp = datetime.now().strftime("%b%d-%Hh%M")
    n_steps_per_epoch = len(train_loader)
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.epoch + 1)
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.n_epochs_per_test == 0:
            test(model, test_ds_path=opt.test_path)
            save(model, opt.save_path, epoch, start_timestamp)
