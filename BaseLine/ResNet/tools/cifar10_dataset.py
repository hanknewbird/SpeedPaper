# -*- coding: utf-8 -*-
import os
from PIL import Image
from torch.utils.data import Dataset


class CifarDataset(Dataset):  # cifar数据集
    def __init__(self, data_dir, transform=None):
        """
        数据集初始化
        :param data_dir: 数据集路径
        :param transform: 对数据集需要处理的transform
        """
        assert (os.path.exists(data_dir)), f"data_dir:{data_dir} 不存在！"  # 当数据集传入错误时报错

        self.data_dir = data_dir    # 将数据集路径作为data_dir属性
        self._get_img_info()        # 将
        self.transform = transform  # 将图像预处理操作保存为transform属性

    def _get_img_info(self):
        """

        :return:
        """
        # 通过循环数据集总路径得到相应的分类后的数据集文件夹名称
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        # 通过总数据集路径+分类后数据集文件夹名称得到分类后的数据集文件夹相对路径
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]

        self.img_info = []  # 创建一个空数组,并设置为子属性,其用来存放
        for c_dir in sub_dir:  # 循环每个分类的文件夹
            # 将每张图像的path和相关label打包
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir))) for i in os.listdir(c_dir) if i.endswith("png")]
            # 将计算得到的path和label附加给img_info属性
            self.img_info.extend(path_img)

    def __len__(self):
        """
        获取数据集长度
        :return: 数据集长度
        """
        if len(self.img_info) == 0:  # 当读取的数据集长度为0时报错
            raise Exception("未获取任何图片路径，请检查dataset及文件路径！")
        return len(self.img_info)

    def __getitem__(self, index):
        """
        传入数据集的index得到图像数据和图像标签
        :param index: 图像索引
        :return: img & label
        """
        fn, label = self.img_info[index]     # 通过索引得到
        img = Image.open(fn).convert('RGB')  # 通过路径以RGB格式读取图像

        if self.transform is not None:
            img = self.transform(img)        # 图像transform操作

        return img, label
