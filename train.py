from utils import clip_gradient, dice
from dataloader import get_train_loader, TestDatasets
from model import PolypSeg, AttentionHead
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import argparse
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

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


def validate(model, test_root, epoch):
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


def train(train_loader, model: PolypSeg, optimizer, epoch):
    model.train()
    attention_head = AttentionHead().cuda()
    attention_head.train()

    for i, batch in enumerate(train_loader, start=1):
        images = batch['image'].cuda()
        gts = batch['mask'].cuda()
        inner_images = batch['inner_image'].cuda()
        slices = batch['slice'][0]

        ## forward
        # inner forward
        inner_output = model(inner_images)

        # outer forward
        x1, x2, x3, x4 = model.encoder(images)
        outer_output = model.segm_head(x1, x2, x3, x4)
        weight_map = attention_head(x1, x2, x3, x4)

        # upscale outer output
        outer_output = F.upsample(outer_output, size=(448, 448), mode='bilinear', align_corners=False)

        inner_output_padded = torch.zeros_like(outer_output)
        weight_map_padded = torch.zeros_like(outer_output)

        x0, y0, x1, y1 = slices.tolist()
        inner_output_padded[:, :, y0:y1, x0:x1] = inner_output
        inner_output_padded[:, :, y0:y1, x0:x1] = weight_map

        # fuse
        output = inner_output_padded * weight_map_padded + outer_output * (1 - weight_map_padded)

        loss = structure_loss(output, gts)

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


def save(model, name, save_path, epoch, start_timestamp):
    os.makedirs(save_path, exist_ok=True)
    file_name = f"{name}.e_{epoch}.{start_timestamp}.pth"
    torch.save(model.state_dict(), os.path.join(save_path, file_name))

def parse_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='name for a training session', default='unnamed')

    parser.add_argument('--epoch', type=int,
                        default=40, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--batchsize', type=int,
                        default=2, help='training batch size')

    parser.add_argument('--train_size', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--n_epochs_per_test', type=float,
                        default=1, help='number of epochs per a test')

    parser.add_argument('--train_roots',
                    default=['./dataset/TrainDataset/'],
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

    return opt

if __name__ == '__main__':
    opt = parse_arg()

    # Model
    model = PolypSeg().cuda()
   
    # Train dataloader
    train_loader = get_train_loader(train_roots=opt.train_roots, batchsize=opt.batchsize, train_size=opt.train_size, num_workers=opt.num_workers)
    n_steps_per_epoch = len(train_loader)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), opt.lr, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[15,25,35], gamma=0.25)

    # Tensorboard writer
    writer = SummaryWriter(f'runs/{opt.name}')

    # Start training
    print("#" * 20, "Start Training", "#" * 20)
    start_timestamp = datetime.now().strftime("%b%d-%Hh%M")
    for epoch in range(1, opt.epoch + 1):
        train(train_loader, model, optimizer, epoch)
        if epoch % opt.n_epochs_per_test == 0:
            validate(model, opt.test_root, epoch)
            save(model, opt.name, opt.save_path, epoch, start_timestamp)
        scheduler.step()
