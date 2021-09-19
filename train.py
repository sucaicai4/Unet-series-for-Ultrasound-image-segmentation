from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import UltraDataset
import model
import utils

if __name__ == "__main__":
    """ 准备数据 """
    train_dataset = UltraDataset("data/train.txt", image_height=512, image_weight=512, image_aug=False)
    val_dataset = UltraDataset("data/val.txt", image_height=512, image_weight=512, image_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)  # , pin_memory=True
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)  # , pin_memory=True

    """ 模型载入 """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = model.Attention_Unet_Vgg(num_class=2, in_channels=3)
    net = net.to(device)

    """ 超参数 """
    log_path = "logs/" + net._get_name() + '/'
    epochs = 20
    total_loss = 0
    avg_loss = 0

    """ 日志记录 """
    logger = utils.logger.Log_loss(log_path)

    best_val_loss = 1000
    val_total_loss = 0
    val_avg_loss = 0

    """ 优化器 """
    loss_func = utils.loss.ce_dice_iou_loss  # CE_Loss()
    optimizer = optim.Adam(params=net.parameters(),
                           lr=0.0001,
                           # momentum=0.949,
                           weight_decay=0.0005
                           )
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

    """ 训练 | 验证 """
    for epoch in range(epochs):
        """ 训练 """
        net.train()
        pbar = tqdm(enumerate(train_loader), desc='Training for %g / %s epoch ' % (epoch + 1, epochs),
                    total=len(train_dataset) // train_loader.batch_size)
        for iteration, (imgs, label) in pbar:
            # 简单处理
            with torch.no_grad():
                imgs = imgs.to(device).float() / 255.0
                label = label.to(device).float().long()

            # 模型推理
            optimizer.zero_grad()
            preds = net(imgs)
            loss = loss_func(preds, label)
            # 反向传播
            loss.backward()
            optimizer.step()

            # caculate the metrics,  acc iou dice
            with torch.no_grad():
                accs, miou, mdice, kappas = utils.calculate_metrics(preds.cpu(), label.cpu())
            # caculate the epoch avg loss
            total_loss += loss.item()
            avg_loss = total_loss / (iteration + 1)

            # 打印信息
            lr = optimizer.param_groups[0]['lr']
            traininfo = "Training: epoch: %d | train loss: %.3f | lr: %.3f | acc: %.3f | miou: %.3f | mdice: %.3f" \
                        % (epoch + 1, avg_loss, lr, accs, miou, mdice)
            pbar.set_description(traininfo)

        # 清零 total_loss
        total_loss = 0

        """ 验证 """
        net.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_dataset) // val_loader.batch_size)
        for iteration, (imgs, label) in pbar:
            with torch.no_grad():
                # 简单处理
                imgs = imgs.to(device).float() / 255.0
                label = label.to(device).float().long()
                # 模型推理
                optimizer.zero_grad()
                preds = net(imgs)

                # caculate the metrics,  acc iou dice
                accs, miou, mdice, kappas = utils.metrics.calculate_metrics(preds.cpu(), label.cpu())
                # caculate the epoch evaluation avg loss
                val_loss = loss_func(preds, label)
                val_total_loss += val_loss.item()
                val_avg_loss = val_total_loss / (iteration + 1)

            # 打印信息
            valinfo = "Evaluating: epoch: %d | val loss: %.3f | lr: %.3f | acc: %.3f | miou: %.3f | mdice: %.3f" \
                      % (epoch + 1, val_avg_loss, lr, accs, miou, mdice)
            pbar.set_description(valinfo)

        # 清零 val_total_loss
        val_total_loss = 0
        # 更新学习率
        scheduler.step(val_avg_loss)
        # 记录日志并绘图
        logger.append_info(traininfo, valinfo)
        logger.append_loss(avg_loss, val_avg_loss)

        """ 保留最优权重 """
        print('finish evaluation')
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            print("Saving state: val loss: %s, in the %s epoch" % (best_val_loss, epoch + 1))
            torch.save(net.state_dict(), log_path + 'epoch%d_total_loss%.4f_val_loss%.4f.pth' %
                       ((epoch + 1), avg_loss, val_avg_loss))
