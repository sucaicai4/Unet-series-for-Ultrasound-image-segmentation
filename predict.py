import os

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import UltraDataset
import model
import utils

from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    """ 准备数据 """
    test_dataset = UltraDataset("data/test.txt", image_height=512, image_weight=512, image_aug=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, pin_memory=True)

    """ 模型载入 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.Attention_Unet_Vgg(num_class=2, in_channels=3)
    net = net.to(device)
    log_path = "logs/" + net._get_name() + '/'
    weight = "epoch19_total_loss0.3155_val_loss0.5044.pth"
    net.load_state_dict(torch.load(log_path + weight))

    """ 预测结果保留 """
    savepath = log_path + "preout/"
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        os.makedirs(savepath + "mask/")

    """ 预测 """
    mean_acc = 0
    mean_dice = 0
    mean_iou = 0
    net.eval()
    pbar = tqdm(enumerate(test_loader), total=len(test_dataset) // test_loader.batch_size)
    for iteration, (imgs, label) in pbar:
        with torch.no_grad():
            # 简单处理
            img_np = imgs.numpy()
            imgs = imgs.to(device).float() / 255.0
            label = label.long()
            # 模型推理
            preds = net(imgs)
            accs, miou, mdice, _ = utils.metrics.calculate_metrics(preds.cpu(), label)
            preds = torch.argmax(preds, dim=1).float()

            preds_np = preds.cpu().numpy()[0]
            label_np = label.numpy()[0]

        imgs_list = [img_np[0][0], preds_np, label_np]
        titles = ["img", "pre", "label"]
        title = "label_%s.png: acc: %.3f | iou: %.3f | dice: %.3f" % (iteration, accs, miou, mdice)
        for i in range(3):
            ax = plt.subplot(1, 3, i + 1)
            plt.xticks([]), plt.yticks([])
            plt.imshow(imgs_list[i], 'gray')
            ax.set_title(titles[i])
        plt.suptitle(title)
        plt.savefig(savepath + "label_%s.png" % iteration)
        # plt.show()
        # cv2.imwrite("preout/pre_%s.png" % iteration, preds_np * 255)
        cv2.imwrite(savepath + "mask/" + "label_%s.png" % iteration, label_np * 255)

        mean_acc += accs
        mean_dice += miou
        mean_iou += mdice

    mean_acc /= iteration + 1
    mean_dice /= iteration + 1
    mean_iou /= iteration + 1

    message = """The prediction results with %s: mean_acc: %.3f | mean_iou: %.3f | mean_dice: %.3f""" \
              % (weight, accs, miou, mdice)
    with open(savepath + "results.txt", 'w') as f:
        f.write(message)
