from utils import clip_gradient, adjust_lr, dice
from dataloader import get_train_loader, TestDatasets
from model import PolypSeg
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import os
import argparse
from datetime import datetime
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


def test(model, test_root, epoch):
    model.eval()
    dice_record = {}
    test_loader = TestDatasets(test_root, 352)
    for ds_name in test_loader.DS_NAMES:
        sum_dice_score = 0.0
        n_imgs = test_loader.datasets[ds_name]["n_imgs"]
        for i in range(n_imgs):
            # Get img, mask
            _, image, gt = test_loader.get_item(ds_name, i)
            image = image.unsqueeze(0).cuda()

            # Infer
            res = model(image)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            
            # Eval dice
            dice_score = dice(res, gt)
            sum_dice_score = sum_dice_score + dice_score

        mean_dice = sum_dice_score / n_imgs
        print(f"{ds_name}: {mean_dice*100:.2f} %")
        dice_record[ds_name] = mean_dice

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

    parser.add_argument('--train_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--n_epochs_per_test', type=float,
                        default=1, help='number of epochs per a test')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_roots',
                    default=['./dataset/TrainDataset/', './dataset/TrainDataset_synthesis/'],
                    nargs='+',
                    help='path to train datasets')

    parser.add_argument('--test_root', type=str,
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
    train_loader = get_train_loader(train_roots=opt.train_roots, batchsize=opt.batchsize, train_size=opt.train_size, num_workers=opt.num_workers)

    # Start training
    print("#" * 20, "Start Training", "#" * 20)
    start_timestamp = datetime.now().strftime("%b%d-%Hh%M")
    n_steps_per_epoch = len(train_loader)
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, opt.epoch + 1)
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.n_epochs_per_test == 0:
            test(model, opt.test_root, epoch)
            save(model, opt.save_path, epoch, start_timestamp)
