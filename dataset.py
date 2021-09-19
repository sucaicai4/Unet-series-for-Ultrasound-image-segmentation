import os
import cv2
import matplotlib.pyplot as plt

from torch import from_numpy
from torch import Tensor
from torch.utils.data import Dataset


class UltraDataset(Dataset):
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(UltraDataset, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask[mask > 1] = 1  # 针对二分类
        mask = from_numpy(mask)
        # mask = mask.unsqueeze(0)

        return from_numpy(image), mask

    def __len__(self):
        return len(self.data_list)


class UltraDataset_ds(Dataset):
    """
    用于 深监督训练 的data generator
    """
    def __init__(self, txt_path, image_height=512, image_weight=512, image_aug=False):
        super(UltraDataset_ds, self).__init__()

        assert os.path.exists(txt_path), "%s 路径有问题！" % txt_path
        with open(txt_path, 'r') as f:
            self.data_list = f.readlines()

        self.image_height = image_height
        self.image_weight = image_weight
        self.image_aug = image_aug

        # 检查txt中的文件是否都存在,可选

    def __getitem__(self, index) -> Tensor:
        image_path, mask_path = self.data_list[index].split()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        if self.image_aug:
            pass
        image = image.transpose(2, 0, 1)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        mask_512 = cv2.resize(mask, (self.image_weight, self.image_height), interpolation=cv2.INTER_LINEAR)
        mask_512[mask_512 > 1] = 1  # 针对二分类

        mask_256 = cv2.resize(mask, (self.image_weight//2, self.image_height//2), interpolation=cv2.INTER_LINEAR)
        mask_256[mask_256 > 1] = 1  # 针对二分类

        mask_128 = cv2.resize(mask, (self.image_weight//4, self.image_height//4), interpolation=cv2.INTER_LINEAR)
        mask_128[mask_128 > 1] = 1  # 针对二分类

        return from_numpy(image), from_numpy(mask_512), from_numpy(mask_256), from_numpy(mask_128)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = UltraDataset_ds("./data/train.txt")
    img, label_512, label_256, label_128 = data.__getitem__(50)

    print(img.shape, label_128.shape)

    # plt.subplot(1, 2, 1)
    # plt.imshow(img.permute(1, 2, 0).numpy(), 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2)
    # plt.imshow(label.numpy() * 255, 'gray')
    # plt.xticks([]), plt.yticks([])
    # plt.show()
