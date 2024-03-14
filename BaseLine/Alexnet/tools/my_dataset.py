# -*- coding: utf-8 -*-
"""
猫狗数据集
"""
import os
import random
from PIL import Image
from torch.utils.data import Dataset


class CatDogDataset(Dataset):
    def __init__(self, data_dir, mode="train", split_n=0.9, rng_seed=620, transform=None):
        """
        分类任务的Dataset

        Args:
            data_dir: 数据集所在路径
            mode: 模式(train/valid)
            split_n: 训练集所占比重
            rng_seed: 置随机种子
            transform: 数据预处理方式
        """
        self.mode = mode                       # train或valid
        self.data_dir = data_dir               # 数据集路径
        self.rng_seed = rng_seed               # 随机种子
        self.split_n = split_n                 # 训练集所占比重
        self.data_info = self._get_img_info()  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform             # 数据预处理方式

    def __getitem__(self, index):
        """
        每次读取都会访问这个函数，不能在这里做数据预处理操作，不然每次都重新预处理了，浪费时间

        Args:
            index: 需要读取的图像索引

        Returns:
            img和label
        """
        path_img, label = self.data_info[index]    # 通过index得到图像路径与标签
        img = Image.open(path_img).convert('RGB')  # 以RGB方式读取图像

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        """
        获取数据集长度

        Returns:
            数据集长度
        """
        if len(self.data_info) == 0:
            raise Exception(f"\n数据集文件夹路径:{self.data_dir}中数据集为空,请检查!!")
        return len(self.data_info)

    def _get_img_info(self):
        """
        获取图片信息

        Returns:完整的图片路径和标签
        """
        img_names = os.listdir(self.data_dir)                              # 获取所有图片的路径
        img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))  # 得到所有jpg图片列表

        random.seed(self.rng_seed)  # 置随机种子,主要是为了以后再次读取的时候得到的图片顺序相同
        random.shuffle(img_names)   # 数据集前后一半都是一样的,所以将图片list打乱,以达到猫狗图片顺序没有规律

        img_labels = [0 if n.startswith('cat') else 1 for n in img_names]  # 如果图片路径以cat开头则标签为0，否则为1

        split_idx = int(len(img_labels) * self.split_n)  # 计算分离索引 25000*0.9=22500
        if self.mode == "train":                # 训练集
            img_set = img_names[:split_idx]     # 训练集路径
            label_set = img_labels[:split_idx]  # 训练集标签
        elif self.mode == "valid":              # 验证集
            img_set = img_names[split_idx:]     # 验证集路径
            label_set = img_labels[split_idx:]  # 验证集标签
        else:
            raise Exception("self.mode无法识别，仅支持(train, valid)")  # 异常报错信息

        path_img_set = [os.path.join(self.data_dir, n) for n in img_set]  # 得到完整的图片路径
        data_info = [(n, l) for n, l in zip(path_img_set, label_set)]     # 得到完整的图片路径和标签

        return data_info
