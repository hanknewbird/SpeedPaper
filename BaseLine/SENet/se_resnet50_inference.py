# -*- coding: utf-8 -*-
import os
import time
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
from tools.se_resnet import get_se_resnet_50
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_transform(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def process_img(path_img):

    # hard code
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std),
    ])

    # path --> img
    img_rgb = Image.open(path_img).convert('RGB')

    # img --> tensor
    img_tensor = img_transform(img_rgb, inference_transform)
    img_tensor.unsqueeze_(0)        # chw --> bchw
    img_tensor = img_tensor.to(device)

    return img_tensor, img_rgb


def load_class_names(p_clsnames, p_clsnames_cn):
    """
    加载标签名
    :param p_clsnames:
    :param p_clsnames_cn:
    :return:
    """
    with open(p_clsnames, "r") as f:
        class_names = json.load(f)
    with open(p_clsnames_cn, encoding='UTF-8') as f:  # 设置文件对象
        class_names_cn = f.readlines()
    return class_names, class_names_cn


if __name__ == "__main__":

    # 预训练文件
    path_state_dict = os.path.join(BASE_DIR, "..", "ModelFile", "seresnet50-60a8950a85b2b.pkl")

    # 需要预测的图片
    path_img = os.path.join(BASE_DIR, "..", "Data", "CatDog", "Golden Retriever from baidu.jpg")
    # path_img = os.path.join(BASE_DIR, "..", "Data", "CatDog", "tiger cat.jpg")

    # index对应的names
    path_classnames = os.path.join(BASE_DIR, "..", "Data", "imagenet1000.json")

    # index对应的中文
    path_classnames_cn = os.path.join(BASE_DIR, "..", "Data", "imagenet_classnames.txt")

    # 载入class names
    cls_n, cls_n_cn = load_class_names(path_classnames, path_classnames_cn)

    # 1/5 加载图片，将图像处理为tensor与rgb两种格式，其中rgb未被处理而是直接读取得到的
    img_tensor, img_rgb = process_img(path_img)

    # 2/5 加载模型，载入预训练模型，并打印模型结构
    se_resnet_model = get_se_resnet_50(path_state_dict, device)

    # 3/5 模型推理
    with torch.no_grad():
        time_tic = time.time()
        outputs = se_resnet_model(img_tensor)
        time_toc = time.time()

    # 4/5 获取类别
    _, pred_int = torch.max(outputs.data, 1)
    _, top5_idx = torch.topk(outputs.data, 5, dim=1)

    pred_idx = int(pred_int.cpu().numpy())
    pred_str, pred_cn = cls_n[pred_idx], cls_n_cn[pred_idx]
    print(f"img: {os.path.basename(path_img)} is: {pred_str}\n{pred_cn}")
    print(f"time consuming:{time_toc - time_tic:.2f}s")

    # 5/5 结果可视化
    plt.imshow(img_rgb)                          # 对图像进行处理，并显示其格式
    plt.title(f"predict:{pred_str}")             # 设置title
    top5_num = top5_idx.cpu().numpy().squeeze()  # 将top_idx从(1,5)拍平为(5,)
    text_str = [cls_n[t] for t in top5_num]      # 找到top_num对应的label name

    # 循环将预测结果展示至画板上
    for idx in range(len(top5_num)):
        plt.text(5, 15+idx*30, f"top {idx + 1}:{text_str[idx]}", bbox=dict(fc='yellow'))

    # 将图像展示出来
    plt.show()
