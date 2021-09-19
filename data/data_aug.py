import cv2
import random
import os
import numpy as np
import time
import math
from tqdm import tqdm


# gamma变化
def gamma_transform(img, gamma):
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table)


def random_gamma_transform(img, mask, gamma_vari):
    # if random.random() < 1:
    log_gamma_vari = np.log(gamma_vari)
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    # gamma = np.exp(alpha)
    # 固定gamma
    gamma = np.exp(log_gamma_vari)
    img = gamma_transform(img, gamma)
    return img, mask


# 模糊原图
def blur(img, mask):
    if random.random() < 0.6:
        img = cv2.blur(img, (3, 3))
    return img, mask


# 添加噪声
def add_noise(img, mask):
    noisenum = random.choice([i for i in range(213, 396)])
    if random.random() < 0.8:
        for i in range(noisenum):  # 添加点噪声
            temp_x = np.random.randint(0, img.shape[0])
            temp_y = np.random.randint(0, img.shape[1])
            img[temp_x][temp_y] = 255
    return img, mask


# 平移
def shift_pic_bboxes(img, mask):
    w = img.shape[1]
    h = img.shape[0]
    d_to_left = int(w / 3)  # 包含所有目标框的最大左移动距离
    d_to_right = int(w / 3)  # 包含所有目标框的最大右移动距离
    d_to_top = int(h / 3)  # 包含所有目标框的最大上移动距离
    d_to_bottom = int(h / 3)  # 包含所有目标框的最大下移动距离

    x = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
    y = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)
    M = np.float32([[1, 0, x],
                    [0, 1, y]])  # x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上

    shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    shift_mask = cv2.warpAffine(mask, M, (img.shape[1], img.shape[0]))
    return shift_img, shift_mask


# 翻转图像
def filp_pic_bboxes(img, mask):
    import copy
    flip_img = copy.deepcopy(img)
    flip_mask = copy.deepcopy(mask)
    if random.random() < 0.5:  # 0.5的概率水平翻转，0.5的概率垂直翻转
        horizon = True
    else:
        horizon = False
    h, w, _ = img.shape
    if horizon:  # 水平翻转
        flip_img = cv2.flip(flip_img, 1)  # 1是水平，-1是水平垂直
        flip_mask = cv2.flip(flip_mask, 1)
    else:
        flip_img = cv2.flip(flip_img, 0)
        flip_mask = cv2.flip(flip_mask, 0)

    return flip_img, flip_mask


def rotate_img_bbox(img, mask, angle=5, scale=1.):
    """
        随机旋转图像和标签图

        Args：
            img(numpy.ndarray): 输入图像
            mask(numpy.ndarray): 标签图
            angle(int)：旋转最大角度，0-90

        Returns：
            旋转后的图像和标签图

        """
    w = img.shape[1]
    h = img.shape[0]
    # 角度变弧度
    rangle = np.deg2rad(angle)  # angle in radians
    # now calculate new image width and height
    nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
    nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
    # ask OpenCV for the rotation matrix
    rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
    # calculate the move from the old center to the new center combined
    # with the rotation
    rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
    # the move only affects the translation, so update the translation
    # part of the transform
    rot_mat[0, 2] += rot_move[0]
    rot_mat[1, 2] += rot_move[1]
    # 仿射变换
    rot_img = cv2.warpAffine(img,
                             rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                             flags=cv2.INTER_LANCZOS4)
    rot_mask = cv2.warpAffine(mask,
                              rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
                              flags=cv2.INTER_LANCZOS4)
    return rot_img, rot_mask


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, mask, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(
            lower, upper)  # 亮度
        img = img / 255.
    return img, mask


def saturation_jitter(cv_img, jitter_range):
    """
    调节图像饱和度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        饱和度调整后的图像

    """

    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def brightness_jitter(cv_img, jitter_range):
    """
    调节图像亮度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        亮度调整后的图像

    """

    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1.0 - jitter_range)
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def contrast_jitter(cv_img, jitter_range):
    """
    调节图像对比度

    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1

    Returns:
        对比度调整后的图像

    """

    greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(greyMat)
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def random_jitter(cv_img, cv_mask, saturation_range, brightness_range, contrast_range):
    """
    图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果

    Args:
        cv_img(numpy.ndarray): 输入图像
        saturation_range(float): 饱和对调节范围，0-1
        brightness_range(float): 亮度调节范围，0-1
        contrast_range(float): 对比度调节范围，0-1

    Returns:
        亮度、饱和度、对比度调整后图像

    """

    # 指定增强
    saturation_ratio = saturation_range
    brightness_ratio = brightness_range
    contrast_ratio = contrast_range

    # 随机增强
    # saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
    # brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
    # contrast_ratio = np.random.uniform(-contrast_range, contrast_range)

    order = [0, 1, 2]
    np.random.shuffle(order)

    for i in range(3):
        if order[i] == 0:
            cv_img = saturation_jitter(cv_img, saturation_ratio)
        if order[i] == 1:
            cv_img = brightness_jitter(cv_img, brightness_ratio)
        if order[i] == 2:
            cv_img = contrast_jitter(cv_img, contrast_ratio)
    return cv_img, cv_mask


if __name__ == '__main__':
    img_path = 'image/'
    mask_path = 'mask/'
    img_all = os.listdir(img_path)
    with open("test.txt", 'r') as f:
        test_list = f.readlines()
    test_list = [item.split()[0].split("e/")[1] for item in test_list]

    train_val = list(set(img_all) - set(test_list))

    for file in tqdm(train_val):
        img0 = cv2.imread(img_path + file)
        mask0 = cv2.imread(mask_path + file.split('.')[0] + '.png')
        for _ in range(8):
            angle_to_rotate = random.choice([i for i in range(0, 20)])  # 产生待旋转角度
            result_img, result_mask = random_gamma_transform(img0, mask0, 1)  # gamma变换
            # result_img, result_mask = random_jitter(img0, mask0, 0, 0, 0)
            # 保留原始图像
            cv2.imwrite('image_aug/' + file, result_img)
            cv2.imwrite('mask_aug/' + file, result_mask)

            result_img, result_mask = rotate_img_bbox(result_img, result_mask, angle=angle_to_rotate)  # 旋转
            result_img, result_mask = shift_pic_bboxes(result_img, result_mask)  # 平移
            result_img, result_mask = filp_pic_bboxes(result_img, result_mask)  # 翻转图像
            result_img, result_mask = add_noise(result_img, result_mask)  # 添加噪声
            result_img, result_mask = blur(result_img, result_mask)  # 随即模糊
            namenew = str(time.time()).split('.')[1] + file.split('.')[0]

            cv2.imwrite('image_aug/' + namenew + '.png', result_img)
            cv2.imwrite('mask_aug/' + namenew + '.png', result_mask)
