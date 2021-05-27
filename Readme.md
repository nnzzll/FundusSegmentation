# 基于UNet的眼底血管分割
## [DRIVE数据](https://drive.grand-challenge.org/)
眼底血管分割,训练集与测试集各20张图像。
下载数据集,解压后得到如下目录形式

    .            -> 根目录
    ├── test     -> 测试集
    │   ├── images
    │   └── mask
    └── training -> 训练集
        ├── 1st_manual
        ├── images
        └── mask
## 环境
<p style="text-indent:2em">os:windows10</p>
<p style="text-indent:2em">python:3.6</p>
<p style="text-indent:2em">cuda:11.0</p>
## 依赖库安装
在终端中运行`install.bat`
    
    install.bat

## 数据处理
直接分割整张图像难度较大，将图像随机crop成小块(patch),再进行训练

    python prepare_dataset.py
参数
- `--patch_per_img`:每张图像切成多少个patch,默认为400
- `--patch_size`:patch的大小,默认为64*64
## 训练
    python train.py
参数

- `--model`:模型名称,目前实现了`UNet`,`UNet++`,`UNet++L3`,`UNet++L2`,`UNet++L1`五种模型

## 测试
    python test.py