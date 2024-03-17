# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from tools.common_tools import get_net


def img_transform(img_l, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_l: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_l)
    return img_t


def process_img(img_path):
    """
    图像预处理
    :param img_path: 输入图像路径
    :return:
    """
    norm_mean = [0.16328026, 0.16328026, 0.16328026]  # 均值
    norm_std = [0.2432042, 0.2432042, 0.2432042]      # 标准差

    inference_transform = transforms.Compose([
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    img_l = Image.open(img_path).convert("RGB")  # 按照指定格式读取图像

    # img --> tensor
    img_t = img_transform(img_l, inference_transform)  # transform
    img_t = img_t.to(device)  # 推至运算设备

    return img_t, img_l


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config
    model_name = f"BaseLine_76e_43.7023%.pkl"
    path_state_dict = os.path.join(BASE_DIR, "results", "BaseLine_P100_B8_test", model_name)

    # 加载模型
    CustomNet = get_net(device=device, vis_model=False, path_state_dict=path_state_dict)
    img_num = 122
    path_img = os.path.join(BASE_DIR, "image", f"{img_num}.jpg")

    # 加载图像
    img_tensor, img_l = process_img(path_img)

    plt.subplot(1, 2, 1)
    plt.imshow(img_l)
    plt.title(f'original')
    plt.axis('off')

    # 预测
    with torch.no_grad():
        outputs = CustomNet(img_tensor.unsqueeze(0))
    mask = outputs.max(dim=1)[1].data.cpu().numpy().squeeze(0)

    # 定义每个类别的颜色映射
    class_colors = {
        0: [0, 0, 0],     # 背景，黑色
        1: [255, 0, 0],   # 折皱，红色
        2: [0, 255, 0],   # 擦伤，绿色
        3: [0, 0, 255],   # 脏污，蓝色
        4: [255, 255, 0]  # 针孔，黄色
    }

    # 创建一个空白的彩色图像
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    # 为每个像素赋予颜色
    for i in range(h):
        for j in range(w):
            colored_mask[i, j] = class_colors[mask[i, j]]

    pre_img = Image.fromarray(colored_mask)
    pre_img.save(f"image/{img_num}_predict.png")

    # 可视化
    plt.subplot(1, 2, 2)
    plt.imshow(pre_img)
    plt.title(f'predict')
    plt.axis('off')
    plt.savefig(f"image/{img_num}_contrast.png")  # 保存图像
    plt.show()
