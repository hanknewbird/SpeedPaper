# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision.models as models


def get_googlenet_v3(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict: 预训练模型
    :param device: 运算设备
    :param vis_model: 是否打印模型结构
    :return: 预训练模型
    """
    model = models.inception_v3()  # 创建模型结构
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict)  # 将预训练模型载入模型

    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchsummary import summary
        summary(model, input_size=(3, 299, 299), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    """
    def __init__(self, eps=0.001):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps

    def forward(self, x, target):
        # CE(q, p) = - sigma(q_i * result(p_i))
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)  # 实现  result(p_i)

        # H(q, p)
        H_pq = -log_probs.gather(dim=-1, index=target.unsqueeze(1))  # 只需要q_i == 1的地方， 此时已经得到CE
        H_pq = H_pq.squeeze(1)

        # H(u, p)
        H_uq = -log_probs.mean()  # 由于u是均匀分布，等价于求均值

        loss = (1 - self.eps) * H_pq + self.eps * H_uq
        return loss.mean()

