

# Ultrasound image segmentation using U-net series 

 <div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/label_0.png" /> </div>

 <div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/label_4.png" /> </div>

 <div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/label_12.png" /> </div>



- ***搭建轻量级 U-net 模型，权重文件 < 15MB,资源占用少***；

- ***所使用的 U-net 模型包括：***

  1. 基础U-net模型，VGG风格Encoder，可迁移PyTorch VGG16模型权重，Maxpooling下采样，UpsamplingBilinear上采样；

  2. 基础U-net模型，VGG风格Encoder，Conv下采样，UpsamplingBilinear上采样；

  3. Attention-Unet，VGG风格Encoder，Maxpooling下采样，UpsamplingBilinear上采样；

     <div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/attunet.jpg" /> </div>

  4. 深监督Attention-Unet，VGG风格Encoder，Maxpooling下采样，UpsamplingBilinear上采样；

   <div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/att_ds.jpg" /> </div>

- ***可在一块 GTX 1050Ti 上进行训练验证；***

## 测试指标结果

|                Model                 | m-Dice | m-IOU | m-Acc | Resolution | Params Size(MB) |
| :----------------------------------: | :----: | :---: | :---: | :--------: | :-------------: |
|       Base U-net (Maxpooling)        | 0.835  | 0.759 | 0.948 |  512*512   |      12.8       |
|          Base U-net (Conv)           | 0.846  | 0.773 | 0.954 |  512*512   |      14.1       |
|            Attention-Unet            | 0.876  | 0.805 | 0.955 |  512*512   |      13.2       |
| Attention-Unet with Deep Supervision | 0.853  | 0.778 | 0.954 |  512*512   |      13.2       |

<div align="center"> <img src="https://github.com/sucaicai4/Unet-series-for-Ultrasound-image-segmentation/blob/main/imgs/models.png" /> </div>

## 数据集

本项目数据集来源为**江苏省生物医学创新设计竞赛**乳腺超声波图像分割赛题，提供带标签数据集**100**张。

链接：https://pan.baidu.com/s/1tEL7gES3Mp_fekOwhup8Tw 
提取码：22v6 

## 模型参考文献

- *Ronneberger, O. , P. Fischer , and T. Brox . "U-Net: Convolutional Networks for Biomedical Image Segmentation." International Conference on Medical Image Computing and Computer-Assisted Intervention Springer International Publishing, 2015.*
- *Oktay, O. , et al. "Attention U-Net: Learning Where to Look for the Pancreas." (CVPR 2018)*
- *Zhou, Z. , et al. "UNet++: A Nested U-Net Architecture for Medical Image Segmentation." 4th Deep Learning in Medical Image Analysis (DLMIA) Workshop 2018.*
- *Abraham, N. , and  N. M. Khan . "A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation." (ISBI 2018).*

## 如何使用

### 安装依赖

- torch==1.9.0
- torchvision==0.10.0
- torchsummary==1.5.1
- tqdm==4.59.0
- opencv-python==4.2.0.34

(低版本库是否兼容未进行验证)

### 下载数据集

下载数据集并解压至 `/data` 文件夹下，文件结构如下所示：

```
/data
├── image
│   ├── breast_001.png
│   ├── breast_002.png
│   └── breast_003.png
└── mask
│   ├── breast_001.png
│   ├── breast_002.png
│   └── breast_003.png
```

运行

```python
python data/test_select.py
```

得到随机挑选的测试集，`test.txt` 如下：

```
data/image/breast_087.png data/mask/breast_087.png
data/image/breast_039.png data/mask/breast_039.png
data/image/breast_044.png data/mask/breast_044.png
```

### 测试

下载 Attention U-net 训练权重，并放至 `/logs/Attention_Unet_Vgg/` 文件夹下：

链接：https://pan.baidu.com/s/1JKgwa0Qup1b9Sk_0gasUHw 
提取码：2s4u 

运行

```python
python predict.py
```

即在 `/logs/Attention_Unet_Vgg/preout` 文件夹下得到预测的标签文件、可视化结果和指标结果。

### 训练

#### 数据准备

在随机挑选完测试集后，运行

```python
python data/data_aug.py
```

**离线**生成数据增强后的图片，文件结构如下：

```
/data
├── image_aug
│   ├── breast_001.png
│   ├── breast_002.png
│   └── breast_003.png
└── mask_aug
│   ├── breast_001.png
│   ├── breast_002.png
│   └── breast_003.png
```

再运行

```python
python data/trainval_split
```

得到划分的训练集 `train.txt`、验证集 `val.txt`，如下

```
data/image_aug/9888062breast_008.png data/mask_aug/9888062breast_008.png
data/image_aug/501469breast_041.png data/mask_aug/501469breast_041.png
data/image_aug/6812515breast_002.png data/mask_aug/6812515breast_002.png
```

```
data/image_aug/breast_004.png data/mask_aug/breast_004.png
data/image_aug/5478067breast_099.png data/mask_aug/5478067breast_099.png
data/image_aug/4450126breast_010.png data/mask_aug/4450126breast_010.png
```

#### 模型选择

目前，可使用的模型包括：

- Base_Unet_Vgg_Simple1
- Base_Unet_Vgg_Simple2
- Attention_Unet_Vgg
- Attention_Unet_ds_Vgg

可以在 `train.py` 文件的 `net = model.Base_Unet_Vgg_Simple1(num_class=2, in_channels=3)` 处配置模型。

配置完成后，运行 `train.py` 即可开始训练，训练过程会被记录在 `logs/model_name` 文件夹下。

### 构建新的模型

可以在 `/model` 文件下搭建其他模型。可用 TersorBoard 对网络进行可视化。

 `/model/arch_plot` 文件夹下是一些网络的 TersorBoard 日志，记录了网络结构，可在该目录下新建终端并运行

```python
tensorboard --logdir="modelname"
```

来查看网络结构。

