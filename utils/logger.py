import os
import datetime
import matplotlib.pyplot as plt
import scipy.signal


class Log_loss:
    def __init__(self, log_dir):
        curr_time = datetime.datetime.now()
        self.time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        self.loss_plot()

    def append_info(self, train_info, val_info):
        with open(os.path.join(self.log_dir, "train_info_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(train_info)
            f.write("\n")
        with open(os.path.join(self.log_dir, "val_info_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_info))
            f.write("\n")

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        num = 5 if len(self.losses) < 25 else 15

        # plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--',
        #          linewidth=2, label="smooth train loss")
        # plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
        #          linewidth=2, label='smooth val loss')

        plt.grid(True)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "loss_" + str(self.time_str) + ".png"))
