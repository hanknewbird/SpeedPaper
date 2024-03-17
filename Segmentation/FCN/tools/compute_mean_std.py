# coding: utf-8
import os
import cv2
import numpy as np

"""
    随机挑选CNum张图片，进行按通道计算均值mean和标准差std
    先将像素从0～255归一化至 0-1 再计算
"""

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径

folder_path = os.path.join('..', '..', 'Data', 'DAGM', 'images', 'train')
# 定义均值和方差
total_mean = 0.0
total_var = 0.0
num_images = 0

# 遍历文件夹中的所有图片
for image_file in os.listdir(folder_path):
    image_path = os.path.join(folder_path, image_file)
    # 读取图片
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # 转换为浮点型数据
    image = image.astype(np.float32) / 255.0
    # 计算均值和方差
    mean = np.mean(image, axis=(0, 1))
    var = np.var(image, axis=(0, 1))
    # 累加均值和方差
    total_mean += mean
    total_var += var
    num_images += 1

# 计算平均值
avg_mean = total_mean / num_images
avg_std = np.sqrt(total_var / num_images)

print('均值:', avg_mean)
print('标准差:', avg_std)
# 均值: [0.16328026 0.16328026 0.16328026]
# 标准差: [0.2432042 0.2432042 0.2432042]
