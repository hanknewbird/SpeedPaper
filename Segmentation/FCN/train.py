import os

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from Segmentation.FCN.tools.dataset import LoadDataset
from tools import FCN
import cfg
from Segmentation.FCN.tools.metrics import *


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')  # 运算设备
num_class = cfg.DATASET[1]  # 类别数量

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)  # 训练集
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)  # 验证集

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)  # 载入训练集
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)  # 载入验证集


fcn = FCN.FCN(num_class)  # 指定网络
fcn = fcn.to(device)  # 将FCN推至运算设备
criterion = nn.NLLLoss().to(device)  # 损失函数
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)  # 优化器


def train(model):
    best = [0]  # 在训练过程中最好的miou值
    train_loss = 0  # 训练损失
    net = model.train()  # 网络开启训练模式
    running_metrics_val = runningScore(12)  # 混淆矩阵

    # 开始训练
    for epoch in range(cfg.EPOCH_NUMBER):
        running_metrics_val.reset()  # 初始化混淆矩阵
        print(f'Epoch is [{epoch + 1}/{cfg.EPOCH_NUMBER}]')

        # 每50轮学习率降低一半
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5

        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))  # [2, 3, 352, 480]
            img_label = Variable(sample['label'].to(device))  # [2, 352, 480]

            # 训练
            out = net(img_data)  # [2, 12, 352, 480]
            out = F.log_softmax(out, dim=1)  # [2, 12, 352, 480]

            loss = criterion(out, img_label)
            optimizer.zero_grad()  # 先梯度清零,再反向传播，最后梯度更新
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()  # 预测标签[2, 1, 352, 480]
            true_label = img_label.data.cpu().numpy()  # 正确标签# [2, 352, 480]
            running_metrics_val.update(true_label, pre_label)  # 更新混淆矩阵

        metrics = running_metrics_val.get_scores()  # 计算各指标

        # 打印各指标
        for k, v in metrics[0].items():
            print(k, v)
        train_miou = metrics[0]['mIou: ']

        # 符合条件后保存模型
        if epoch > 160:
            if max(best) <= train_miou:
                best.append(train_miou)
                t.save(net.state_dict(), os.path.join(cfg.OUTPUT_MODEL_ROOT, f"{epoch}.pth"))


def evaluate(model):
    """
    验证模型
    :param model: 模型
    :return:
    """

    net = model.eval()  # 模型
    running_metrics_val = runningScore(12)  # 混淆矩阵
    eval_loss = 0  # 验证损失
    prec_time = datetime.now()  # 时间

    for j, sample in enumerate(val_data):  # 载入dataloader
        # 取出图像与标签
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        # 预测
        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = valLabel.data.cpu().numpy()
        running_metrics_val.update(true_label, pre_label)
    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = f'Time: {h:.0f}:{m:.0f}:{s:.0f}'
    print(time_str)


if __name__ == "__main__":
    train(fcn)
