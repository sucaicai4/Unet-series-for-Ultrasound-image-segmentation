import torch
import torch.nn as nn
from torchkeras import summary

import numpy as np


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 1000, pretrain: bool = True) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.pretrain = pretrain
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        print("initialize weights...")
        if self.pretrain:
            print("initialize the pretained weights...")

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_dict = self.state_dict()
            pretrained_dict = torch.load("backbone/vgg16-397923af.pth", map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            # self.load_state_dict(torch.load("backbone/vgg16-397923af.pth"), strict=False)   # model/
        else:
            print("initialize the random weights...")
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels=3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=(3, 3), padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {'D': [64, 64, 'M', 128, 128, 'M', 256, 256,
              256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']}


def VGG16(in_channels=3, pretrain=False, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm=True, in_channels=in_channels), pretrain=pretrain, **kwargs)
    return model.features[0:43]


if __name__ == "__main__":
    test_model = VGG16()
    print(test_model)
    summary(test_model,  (3, 512, 512))
