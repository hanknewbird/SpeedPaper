# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from tools.model_alexnet import AlexNet

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 基础路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备

if __name__ == "__main__":

    log_dir = os.path.join(BASE_DIR, "", "result")  # 输出路径
    # ----------------------------------- 卷积核可视化 -----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_kernel")                 # 创建一个Summary文件,并定义后缀
    path_state_dict = os.path.join(BASE_DIR, "ModelFile", "alexnet-owt-4df8aa71.pth")  # 预训练模型路径
    alexnet = AlexNet()                                                                # 定义模型
    pretrained_state_dict = torch.load(path_state_dict)                                # 载入预训练模型
    alexnet.load_state_dict(pretrained_state_dict)                                     # 将模型载入预训练模型参数

    kernel_num = -1  # 初始化卷积层数量为-1
    vis_max = 1      # 最大卷积层数量

    for sub_module in alexnet.modules():               # 遍历alexnet的所有模块
        if not isinstance(sub_module, nn.Conv2d):      # 判断是否是卷积层
            continue
        kernel_num += 1                                # 卷积层数量加1
        if kernel_num > vis_max:                       # 判断是否超过最大卷积层数量
            break                                      # 超过则退出循环

        kernels = sub_module.weight                    # 取出卷积层的权重，其实就是卷积核
        c_out, c_in, k_h, k_w = tuple(kernels.shape)   # 将卷积核拆分为4部分,分别为o,i,h,w,为64,3,11,11

        # 拆分channel
        for o_idx in range(c_out):
            kernel_idx = kernels[o_idx, :, :, :].unsqueeze(1)                                       # 获得(3, h, w), 但是make_grid需要BCHW，这里拓展C维度变为（3，1，h,w）
            kernel_grid = vutils.make_grid(kernel_idx, normalize=True, scale_each=True, nrow=c_in)  # 将卷积核保存为grid格式
            writer.add_image(f'{kernel_num}_通道中分离的卷积层', kernel_grid, global_step=o_idx)    # 写入磁盘

        kernel_all = kernels.view(-1, 3, k_h, k_w)                                    # 3, h, w
        kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=8)  # c, h, w
        writer.add_image(f'{kernel_num}_所有', kernel_grid, global_step=620)

        print(f"{kernel_num}卷积层形状:{tuple(kernels.shape)}")

    # ----------------------------------- 特征图可视化 -----------------------------------
    writer = SummaryWriter(log_dir=log_dir, filename_suffix="_feature map")

    # 数据
    path_img = os.path.join(BASE_DIR, "..", "Data", "tiger cat.jpg")
    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)  # chw --> bchw

    # 前向传播
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0, 1)  # bchw=(1, 64, 55, 55) --> (64, 1, 55, 55)
    fmap_1_grid = vutils.make_grid(fmap_1, normalize=True, scale_each=True, nrow=8)

    writer.add_image('conv1中的feature map', fmap_1_grid, global_step=620)
    writer.close()
