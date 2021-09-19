import os
import random

if __name__ == '__main__':
    if not (os.path.exists('image') and os.path.exists('mask')):
        raise FileNotFoundError("请放置数据集文件夹!")

    image_list = os.listdir('image')
    mask_list = os.listdir('mask')

    assert len(image_list) == len(mask_list), "数据和标签数量不一致!"

    img_mask = list(zip(image_list, mask_list))
    random.shuffle(img_mask)

    with open("test.txt", 'w') as f:
        for image, mask in img_mask[int(0.85 * len(img_mask)):]:
            f.writelines("data/image/%s data/mask/%s\n" % (image, mask))
    f.close()
