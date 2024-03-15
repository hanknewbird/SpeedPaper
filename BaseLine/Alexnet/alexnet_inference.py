# -*- coding: utf-8 -*-
import json
import os
import time
import matplotlib
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.model_alexnet import AlexNet
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 设置base路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断硬件环境


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式

    Args:
        img_rgb: 一张RGB图像
        transform: 自定义的torchvision.transform顺序

    Returns:
        预处理后的tensor格式图像
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名

    Args:
        p_clsnames: 英文标签路径
        p_clsnames_cn: 中文标签路径

    Returns:
        英文标签，中文标签
    """

    with open(p_clsnames, "r") as f:
        class_names = json.load(f)      # 载入json格式的name
    with open(p_clsnames_cn, encoding='UTF-8') as f:
        class_names_cn = f.readlines()  # 载入中文
    return class_names, class_names_cn


def get_model(path_state_dict, vis_model=False):
    """
    载入模型

    Args:
        path_state_dict: 预训练模型路径
        vis_model: 是否打印模型结构

    Returns:
        预训练模型
    """

    model = AlexNet()                                    # 实例化模型
    pretrained_state_dict = torch.load(path_state_dict)  # 载入预训练模型参数
    model.load_state_dict(pretrained_state_dict)         # 将参数载入模型
    model.eval()                                         # 设置为预测模式

    if vis_model:
        from torchsummary import summary
        # 打印网络结构图
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)  # 将模型推至到cpu或gpu上
    return model


def process_img(path_img):
    """
    图片预处理

    Args:
        path_img: 图片的路径

    Returns:
        返回tensor和rgb格式的image
    """

    # 硬编码
    norm_mean = [0.485, 0.456, 0.406]  # imagenet上计算得到的图像均值
    norm_std = [0.229, 0.224, 0.225]   # imagenet上计算得到的图像方差
    inference_transform = transforms.Compose([
        transforms.Resize(256),                     # 将图像的尺寸调整为256*256像素
        transforms.CenterCrop((224, 224)),          # 中心裁剪出224*224像素的区域
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 将图像数据标准化到均值为0、标准差为1的分布
    ])
    # 常见疑问：在ToTensor阶段不是已经归一化到0-1了吗？为什么还要Normalize？
    """
    在处理图像数据时，我们经常先将图像转化为Tensor，然后再进行归一化。
    "ToTensor" 将图像数组的像素值从0-255调整为0-1的尺度，保证模型处理的数据范围统一，有利于模型的训练。
    "Normalize" 是另一个过程，主要目的是将数据的分布标准化，即使得数据集的均值为0，标准差为1。这是从统计学角度出发的，这样做有两个主要好处：
        1.加速模型的收敛：如果数据的分布不平衡，即某些特征的范围远大于其他特征，那么可能会导致梯度下降时陷入局部最小值。
          归一化可以解决这个问题，因为标准化后的数据集收敛到解决方案的速度更快。
        2.提高模型的精度：归一化后，由于所有特征的范围都已经标准化，因此可能会降低模型复杂度，而这样可能会提高模型的精度。
    所以，"Normalize" 会接在 "ToTensor" 后面，起到进一步归一化的作用。
    """

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')  # 以RGB格式读取图像

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)  # 图像预处理
    img_tensor.unsqueeze_(0)                                  # chw --> bchw  扩展b的维度
    img_tensor = img_tensor.to(device)                        # 将tensor放置到运算设备上

    return img_tensor, img_rgb


if __name__ == "__main__":

    # 预训练文件
    path_state_dict = os.path.join(BASE_DIR, "..", "ModelFile", "alexnet-owt-4df8aa71.pth")

    # 需要预测的图片
    path_img = os.path.join(BASE_DIR, "..", "Data", "CatDog", "tiger cat.jpg")
    # path_img = os.path.join(BASE_DIR, "..", "Data", "CatDog", "Golden Retriever from baidu.jpg")

    # index对应的names
    path_classnames = os.path.join(BASE_DIR, "..", "Data", "imagenet1000.json")

    # index对应的中文
    path_classnames_cn = os.path.join(BASE_DIR, "..", "Data", "imagenet_classnames.txt")

    # 载入class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 加载图片，将图像处理为tensor与rgb两种格式，其中rgb未被处理而是直接读取得到的
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 加载模型，载入预训练模型，并打印模型结构
    alexnet_model = get_model(path_state_dict, True)

    # 3/5 模型推理
    with torch.no_grad():                    # 不需要更新梯度
        time_tic = time.time()               # 记录开始预测时间
        outputs = alexnet_model(img_tensor)  # 将传入tensor格式的图像预测成(1,1000)格式的输出结果
        time_toc = time.time()               # 记录结束预测时间

    # 4/5 获取类别
    _, pred_int = torch.max(outputs.data, 1)          # 获取top1的值与相应索引
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)  # 获取top5的值与相应索引值

    pred_idx = int(pred_int.cpu().numpy()[0])                # 将tensor转化为int格式
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]  # 根据索引获取对应的name与中文名称

    # 打印相关信息
    print(f"img: {os.path.basename(path_img)} is: {pred_str}\n{pred_cn}")
    print(f"time consuming:{time_toc - time_tic:.2f}s")

    # 5/5 结果可视化
    plt.imshow(img_rgb)                          # 对图像进行处理，并显示其格式
    plt.title("predict:{}".format(pred_str))     # 设置title
    top5_num = top5_idx.cpu().numpy().squeeze()  # 将top_idx从(1,5)拍平为(5,)
    text_str = [cls_n[t] for t in top5_num]      # 找到top_num对应的label name

    # 循环将预测结果展示至画板上
    for idx in range(len(top5_num)):
        plt.text(5, 15 + idx * 30, f"top {idx + 1}:{text_str[idx]}", bbox=dict(fc='yellow'))

    # 将图像展示出来
    plt.show()
