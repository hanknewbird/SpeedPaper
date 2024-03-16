import pandas as pd
import os
import torch as t
import numpy as np
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cfg
from torch.utils.data import DataLoader


class LabelProcessor:
    """
        对标签图像的编码
    """

    def __init__(self, file_path):
        """
        初始化
        :param file_path: label路径
        """

        self.colormap = self.read_color_map(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    # 静态方法装饰器,可以理解为定义在类中的普通函数,可以用self.<name>方式调用
    # 在静态方法内部不可以示例属性和实列对象,即不可以调用self.相关的内容
    # 使用静态方法的原因之一是程序设计的需要（简洁代码,封装功能等）
    @staticmethod
    def read_color_map(file_path):
        """
        传入class_dict.csv路径,计算得到对应colormap
        :param file_path: class_dict.csv文件路径
        :return:colormap
        """

        pd_label_color = pd.read_csv(file_path, sep=',')  # 读取class_dict.csv
        colormap = []  # colormap文件
        for i in range(len(pd_label_color.index)):  # 遍历colormap
            tmp = pd_label_color.iloc[i]  # 按行读取
            color = [tmp['r'], tmp['g'], tmp['b']]  # 将colormap分为R,G,B
            colormap.append(color)  # 将[R,G,B]纳入colormap
        return colormap

    @staticmethod
    def encode_label_pix(colormap):
        """
        将colormap(RGB)传入,得到相关hash表,提高查找效率
        其实就是：形成颜色到标签的一一对应关系
        :param colormap: 标签编码
        :return: 返回哈希表
        """

        cm2lbl = np.zeros(256 ** 3)  # 定义一个足够大小的容器存储hash表
        for i, cm in enumerate(colormap):
            # hash = (R*N + G) * N + B
            # 由RGB转化为hash表结构
            cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i
        return cm2lbl

    def encode_label_img(self, img):
        """
        矩阵化批量操作像素点的编码由（r, g, b） ---> index ---> identity
        :param img: 图像
        :return: 由索引带出来的一片表示类别的数值
        """

        data = np.array(img, dtype='int32')
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(self.cm2lbl[idx], dtype='int64')


class LoadDataset(Dataset):
    def __init__(self, file_path=None, crop_size=None):
        """
        :param file_path: 数据和标签路径,列表元素第一个为图片路径,第二个为标签路径
        :param crop_size: 裁剪大小
        """

        if file_path is None:
            file_path = []
        if len(file_path) != 2:
            raise ValueError("同时需要图片和标签文件夹的路径,图片路径在前")

        # 1 正确读入图片和标签路径
        self.img_path = file_path[0]
        self.label_path = file_path[1]

        # 2 从路径中取出图片和标签数据的文件名保持到两个列表当中（程序中的数据来源）
        self.imgs = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

        # 3 初始化数据处理函数设置
        self.crop_size = crop_size

    def __getitem__(self, index):
        """
        根据索引获取数据
        :param index: 索引
        :return: 数据格式为{img, label}
        """

        # 根据index从imgs和labels获取对应的图片和标签
        img = self.imgs[index]
        label = self.labels[index]

        # 读取数据（图片和标签都是png格式的图像数据）
        img = Image.open(img)
        label = Image.open(label).convert('RGB')

        img, label = self.center_crop(img, label, self.crop_size)  # 裁剪尺寸
        img, label = self.img_transform(img, label)  # 图片进行transform

        return {'img': img, 'label': label}

    def __len__(self):
        """
        :return: 数据集长度
        """

        return len(self.imgs)

    def read_file(self, path):
        """
        从文件夹中读取数据
        :param path: 文件夹路径
        :return: 排序后的文件路径
        """

        files_list = os.listdir(path)  # 获取路径下所有文件
        file_path_list = [os.path.join(path, img) for img in files_list]  # 得到所有文件的路径
        file_path_list.sort()  # 排序
        return file_path_list

    def center_crop(self, data, label, crop_size):
        """
        裁剪输入的图片和标签大小
        :param data: 图片
        :param label: 标签
        :param crop_size: 中心裁剪后的大小
        :return: 裁剪后的图片和标签
        """

        # 中心裁剪
        data = ff.center_crop(data, crop_size)
        label = ff.center_crop(label, crop_size)
        return data, label

    def img_transform(self, img, label):
        """
        对图片和标签做一些数值处理
        :param img: 图片
        :param label: 标签
        :return: 进行transform操作后的图片和标签
        """

        label = np.array(label)  # 以免不是np格式的数据
        label = Image.fromarray(label.astype('uint8'))

        # 定义transform操作过程
        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),  # 转化为tensor格式
                transforms.Normalize(  # 进行归一化
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )

        img = transform_img(img)  # transform操作
        label = label_processor.encode_label_img(label)  # 编码
        label = t.from_numpy(label)  # 转化为tensor格式

        return img, label


label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == "__main__":
    Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    train_data = DataLoader(Load_train, batch_size=1, shuffle=True, num_workers=1)
    for sample in train_data:
        img_data = sample['img']
        img_label = sample['label']
        print(img_data.shape)
        print(img_label.shape)
