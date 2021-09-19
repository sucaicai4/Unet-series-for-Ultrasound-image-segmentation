import torch
import torch.nn as nn
from torch.autograd import Variable

from torchkeras import summary
from model.backbone.vgg16 import VGG16
from torch.utils.tensorboard import SummaryWriter


class Unet_Skip_Up(nn.Module):
    """
    Unet_Skip_Up 是 Unet 跳跃链接模块
    """
    def __init__(self, in_channel, out_channel):
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mid_channel = (self.out_channel + self.out_channel) // 2
        super(Unet_Skip_Up, self).__init__()
        self.conv_relu_2 = nn.Sequential(
            nn.Conv2d(self.in_channel, self.mid_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.mid_channel),
            nn.ReLU(),
            nn.Conv2d(self.mid_channel, self.out_channel, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(self.out_channel),
            nn.ReLU()
        )
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, feat_encoder, feat_up):
        feat_up = self.upsample(feat_up)
        output_feat = torch.cat([feat_encoder, feat_up], dim=1)
        output_feat = self.conv_relu_2(output_feat)
        return output_feat


class Base_Unet_Vgg_Official(nn.Module):
    """
    Base_Unet_Vgg_Official 是基于 torchvision 中 vgg16 网络的基础 Unet 模型，方便迁移学习；

    parameter: num_class: 默认二分类
    parameter: in_channels: 默认彩图
    parameter: pretrain: 是否进行迁移学习

    下采样： Maxpooling
    上采样： UpsampleBilinear
    """

    def __init__(self, num_class=2, in_channels=3, pretrain=False):
        super(Base_Unet_Vgg_Official, self).__init__()
        self.num_class = num_class
        print(self.__doc__)
        self.encoder = VGG16(in_channels=in_channels, pretrain=pretrain)
        self.skip_up_64 = Unet_Skip_Up(1024, 256)
        self.skip_up_128 = Unet_Skip_Up(512, 128)
        self.skip_up_256 = Unet_Skip_Up(256, 64)
        self.skip_up_512 = Unet_Skip_Up(128, 32)
        self.cls_conv = nn.Conv2d(32, self.num_class, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        feat_512 = self.encoder[0:6](x)
        feat_256 = self.encoder[6:13](feat_512)
        feat_128 = self.encoder[13:23](feat_256)
        feat_64 = self.encoder[23:33](feat_128)
        feat_32 = self.encoder[33:43](feat_64)

        up_64 = self.skip_up_64(feat_64, feat_32)
        up_128 = self.skip_up_128(feat_128, up_64)
        up_256 = self.skip_up_256(feat_256, up_128)
        up_512 = self.skip_up_512(feat_512, up_256)
        results = self.cls_conv(up_512)

        return results


if __name__ == "__main__":
    base_unet_vgg_official = Base_Unet_Vgg_Official(num_class=2, in_channels=3, pretrain=False)
    summary(base_unet_vgg_official, (3, 512, 512))

    # dummy_input = torch.randn(1, 3, 512, 512)
    # outputs = base_unet_vgg_official(dummy_input)
    # print(outputs.shape)

    # writer = SummaryWriter("arch_plot/" + base_unet_vgg_official._get_name())
    # writer.add_graph(base_unet_vgg_official, torch.randn((1, 3, 512, 512)))
    # writer.close()
