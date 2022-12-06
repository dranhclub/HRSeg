import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.bn(x)
        return x

class LocalEmphasis(nn.Module):
    def __init__(self, in_planes, out_planes, up_sample_scale_factor=2):
        super().__init__()
        self.conv1 = BasicConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=up_sample_scale_factor, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class PolypSeg(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.LE4 = LocalEmphasis(in_planes=512, out_planes=512, up_sample_scale_factor=8)
        self.LE3 = LocalEmphasis(in_planes=320, out_planes=320, up_sample_scale_factor=4)
        self.LE2 = LocalEmphasis(in_planes=128, out_planes=128, up_sample_scale_factor=2)
        self.LE1 = LocalEmphasis(in_planes=64, out_planes=64, up_sample_scale_factor=1)
        
        self.linear34 = nn.Conv2d(in_channels=320+512, out_channels=320, kernel_size=1)
        self.linear23 = nn.Conv2d(in_channels=320+128, out_channels=128, kernel_size=1)
        self.linear12 = nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=1)
        self.linear_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)


    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)

        x4 = self.LE4(x4)
        x3 = self.LE3(x3)
        x2 = self.LE2(x2)
        x1 = self.LE1(x1)

        x34 = torch.cat([x3, x4], dim=1)
        x34 = self.linear34(x34)
        x23 = torch.cat([x2, x34], dim=1)
        x23 = self.linear23(x23)
        x12 = torch.cat([x1, x23], dim=1)
        x12 = self.linear12(x12)

        x_out = self.linear_out(x12)
        x_out = F.interpolate(x_out, scale_factor=4, mode='bilinear')

        return x_out

if __name__ == "__main__":
    model = PolypSeg()

    from thop import profile, clever_format
    inputs = torch.rand(1, 3, 224, 224)
    print(clever_format(profile(model, inputs=(inputs, ))))