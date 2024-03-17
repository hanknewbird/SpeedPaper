# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train', img_transform=None, label_transform=None):
        assert (os.path.exists(data_dir)), f"data_dir:{data_dir} 不存在！"

        self.data_dir = data_dir
        self.mode = mode
        self._get_img_info()
        self.img_transform = img_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img_path, label_path = self.img_info[index]

        img = Image.open(img_path).convert("RGB")  # RGB图
        img = self.img_transform(img)

        label = Image.open(label_path).convert("L")  # 灰度图
        # label = self.label_transform(label)
        label = np.array(label)
        label = torch.tensor(label, dtype=torch.int8)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        """获取图像路径和目标图像路径"""

        # 图像根路径和标签根路径
        images_dir = os.path.join(self.data_dir, 'images', self.mode)
        labels_dir = os.path.join(self.data_dir, 'masks', self.mode)

        assert os.path.exists(images_dir), f"{images_dir}不存在！"
        assert os.path.exists(labels_dir), f"{labels_dir}不存在！"

        label_paths = os.listdir(labels_dir)

        self.img_info = []

        # img, label
        path_img = [(os.path.join(images_dir, i.replace("png", "jpg")), os.path.join(labels_dir, i)) for i in label_paths if i.endswith("png")]

        self.img_info.extend(path_img)
