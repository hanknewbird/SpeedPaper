# -*- coding: utf-8 -*-
"""
# @file name  : common_tools.py
# @brief      : 通用函数
"""
import torch
import torchvision.models as models


def get_vgg16(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict: 预训练模型
    :param device: 运算设备
    :param vis_model: 是否打印模型结构
    :return: 预训练模型
    """
    model = models.vgg16()  # 创建模型结构
    pretrained_state_dict = torch.load(path_state_dict)  # 读取预训练模型
    model.load_state_dict(pretrained_state_dict)  # 将预训练模型载入模型
    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model
