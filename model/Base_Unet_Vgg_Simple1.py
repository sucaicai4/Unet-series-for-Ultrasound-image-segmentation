import torch
import torch.nn as nn
from torch.autograd import Variable

from torchkeras import summary
from torch.utils.tensorboard import SummaryWriter


class Double_CBR(nn.Module):
    """
    Double_CBR 是 Conv BN Relu 两次堆叠
    """
    def __init__(self, in_channel, out_channel, is_pooling=False):
        super(Double_CBR, self).__init__()
        self.is_pooling = is_pooling
        self.CBR2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6()
        )
        if self.is_pooling:
            self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        if self.is_pooling:
            x = self.pool(x)
        x = self.CBR2(x)
        return x


class Unet_Skip_Up(nn.Module):
    """
    Unet_Skip_Up 是 Unet 跳跃链接模块
    """
    def __init__(self, in_channel, out_channel):
        super(Unet_Skip_Up, self).__init__()
        self.CBR2 = Double_CBR(in_channel, out_channel, is_pooling=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat_encoder, feat_up):
        feat_up = self.upsample(feat_up)
        output_feat = torch.cat([feat_encoder, feat_up], dim=1)
        output_feat = self.CBR2(output_feat)
        return output_feat


class Unet_Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super(Unet_Encoder, self).__init__()
        self.CBR_512 = Double_CBR(in_channel, 32, is_pooling=False)
        self.CBR_256 = Double_CBR(32, 64, is_pooling=True)
        self.CBR_128 = Double_CBR(64, 128, is_pooling=True)
        self.CBR_64 = Double_CBR(128, 256, is_pooling=True)
        self.CBR_32 = Double_CBR(256, 256, is_pooling=True)

    def forward(self, x):
        feat_512 = self.CBR_512(x)
        feat_256 = self.CBR_256(feat_512)
        feat_128 = self.CBR_128(feat_256)
        feat_64 = self.CBR_64(feat_128)
        feat_32 = self.CBR_32(feat_64)

        return feat_512, feat_256, feat_128, feat_64, feat_32


class Base_Unet_Vgg_Simple1(nn.Module):
    """
    Base_Unet_Vgg_Simple 是基于 Conv BN Relu6 堆叠的 Vgg 风格的简单 Unet

    parameter: num_class: 默认二分类
    parameter: in_channels: 默认彩图

    下采样： Maxpooling
    上采样： UpsampleBilinear
    """
    def __init__(self, num_class=2, in_channels=3):
        super(Base_Unet_Vgg_Simple1, self).__init__()
        print(self.__doc__)
        self.encoder = Unet_Encoder(3)
        self.skip_up_64 = Unet_Skip_Up(512, 128)
        self.skip_up_128 = Unet_Skip_Up(256, 64)
        self.skip_up_256 = Unet_Skip_Up(128, 32)
        self.skip_up_512 = Unet_Skip_Up(64, 32)
        self.cls_conv = nn.Conv2d(32, num_class, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        feat_512, feat_256, feat_128, feat_64, feat_32 = self.encoder(x)

        up_64 = self.skip_up_64(feat_64, feat_32)
        up_128 = self.skip_up_128(feat_128, up_64)
        up_256 = self.skip_up_256(feat_256, up_128)
        up_512 = self.skip_up_512(feat_512, up_256)

        results = self.cls_conv(up_512)
        return results


if __name__ == "__main__":
    base_unet_vgg_simple = Base_Unet_Vgg_Simple1(num_class=2, in_channels=3)
    summary(base_unet_vgg_simple, (3, 512, 512))

    # dummy_input = torch.randn(1, 3, 512, 512)
    # outputs = base_unet_vgg_official(dummy_input)
    # print(outputs.shape)

    # writer = SummaryWriter("arch_plot/" + base_unet_vgg_simple._get_name())
    # writer.add_graph(base_unet_vgg_simple, torch.randn((1, 3, 512, 512)))
    # writer.close()
