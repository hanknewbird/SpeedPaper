# -*- coding: utf-8 -*-
import os
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from matplotlib import pyplot as plt
from tools.my_dataset import NCFMDataSet
import torchvision.models as models


def get_googlenet(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict: 预训练模型
    :param device: 运算设备
    :param vis_model: 是否打印模型结构
    :return: 预训练模型
    """
    model = models.googlenet(init_weights=False)  # 创建模型结构
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict)  # 将预训练模型载入模型

    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备


if __name__ == "__main__":

    # 参数配置  如果缺少数据集或预训练文件，请阅读相关readme获取下载地址
    # 数据集："https://github.com/hanknewbird/SpeedPaper/tree/main/BaseLine/Data"
    # 预训练文件："https://github.com/hanknewbird/SpeedPaper/tree/main/BaseLine/ModelFile"
    data_dir = os.path.join(BASE_DIR, "..", "Data", "NCFM", "train")                       # 基础路径
    path_state_dict = os.path.join(BASE_DIR, "..", "ModelFile", "googlenet-1378be20.pth")  # 以googlenetv1的预训练模型参数路径

    num_classes = 8    # 定义类别
    MAX_EPOCH = 3      # 跑多少轮
    BATCH_SIZE = 8   # 每次载入多少图片
    LR = 0.001         # 学习率
    log_interval = 1   # 多少次打印一次log
    val_interval = 1   # 多少次验证一次
    start_epoch = -1   # 开始的epoch(断点训练时有用)
    lr_decay_step = 5  # 学习率衰减步长

    # ============================ step 1/5 数据 ============================
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([          # 定义图片的预训练方式
        transforms.Resize(256),                     # (256)是短边压缩，注意与(256, 256)的区别
        transforms.CenterCrop(256),                 # 中心裁剪为(256,256)
        transforms.RandomCrop(224),                 # 随机裁剪为(224,224)
        transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    normalizes = transforms.Normalize(norm_mean, norm_std)
    valid_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 裁剪为(256,256)
        transforms.TenCrop(224, vertical_flip=False),  # 取10份(垂直翻转)
        # 通过ToTensor和normalizes处理后将10张图片链接起来
        transforms.Lambda(lambda crops: torch.stack([normalizes(transforms.ToTensor()(crop)) for crop in crops])),
    ])

    # 构建MyDataset实例
    train_data = NCFMDataSet(data_dir=data_dir, mode="train", transform=train_transform)
    valid_data = NCFMDataSet(data_dir=data_dir, mode="valid", transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=4)

    # ============================ step 2/5 模型 ============================
    googlenet_model = get_googlenet(path_state_dict, device, False)

    num_ftrs = googlenet_model.fc.in_features
    googlenet_model.fc = nn.Linear(num_ftrs, num_classes)

    num_ftrs_1 = googlenet_model.aux1.fc2.in_features
    googlenet_model.aux1.fc2 = nn.Linear(num_ftrs_1, num_classes)

    num_ftrs_2 = googlenet_model.aux2.fc2.in_features
    googlenet_model.aux2.fc2 = nn.Linear(num_ftrs_2, num_classes)

    googlenet_model.to(device)
    # ============================ step 3/5 损失函数 ============================

    criterion = nn.CrossEntropyLoss()

    # ============================ step 4/5 优化器 ============================
    # 冻结卷积层
    flag = 0
    # flag = 1
    if flag:
        fc_params_id = list(map(id, googlenet_model.classifier.parameters()))  # 返回的是parameters的 内存地址
        base_params = filter(lambda p: id(p) not in fc_params_id, googlenet_model.parameters())
        optimizer = optim.SGD([
            {'params': base_params, 'lr': LR * 0.1},  # 0
            {'params': googlenet_model.classifier.parameters(), 'lr': LR}], momentum=0.9)

    else:
        optimizer = optim.SGD(googlenet_model.parameters(), lr=LR, momentum=0.9)  # 选择优化器

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)  # 设置学习率下降策略
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(patience=5)

# ============================ step 5/5 训练 ============================

    train_curve = list()  # 训练曲线
    valid_curve = list()  # 验证曲线

    for epoch in range(start_epoch + 1, MAX_EPOCH):  # 开始训练模型

        loss_mean = 0.  # 平均loss
        correct = 0.  # 正确预测数量
        total = 0.  # 总共数量

        googlenet_model.train()  # 模型开启训练模式
        for i, data in enumerate(train_loader):  # 对训练集进行迭代,i为第几次,data为数据

            # 前向计算
            inputs, labels = data  # 获取RGB格式的图像和对应标签
            inputs, labels = inputs.to(device), labels.to(device)  # 分别推至运算设备上
            outputs = googlenet_model(inputs)  # 预测输出

            # 反向传播
            optimizer.zero_grad()  # 梯度清零
            loss_main = criterion(outputs[0], labels)  # 计算损失
            aug_loss1 = criterion(outputs[1], labels)  # 辅助损失
            aug_loss2 = criterion(outputs[2], labels)  # 辅助损失
            loss = loss_main + (0.3 * aug_loss1) + (0.3 * aug_loss2)
            loss.backward()  # 反向传播

            # 更新权重
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs[0].data, 1)  # 获取top1的值与相应索引
            total += labels.size(0)  # 记录总数
            correct += (predicted == labels).squeeze().cpu().sum().numpy()  # 预测正确数量

            # 打印训练信息
            loss_mean += loss_main.item()  # 获取loss的值
            train_curve.append(loss_main.item())  # 将本次的loss值加入到训练曲线列表中
            if (i+1) % log_interval == 0:  # 是否打印日志
                loss_mean = loss_mean / log_interval  # 计算loss平均值
                print(f"Training:Epoch[{epoch:0>3}/{MAX_EPOCH:0>3}] "
                      f"Iteration[{i+1:0>3}/{len(train_loader):0>3}] "
                      f"Loss_main:{loss_mean:.4f} "
                      f"Acc:{correct / total:.2%} "
                      f"lr:{scheduler.get_last_lr()}")
                loss_mean = 0.

        scheduler.step()  # 更新学习率

        # 验证模式
        if (epoch+1) % val_interval == 0:

            correct_val = 0.  # 验证正确数量
            total_val = 0.  # 总数
            loss_val = 0.  # loss值

            googlenet_model.eval()  # 模型开启验证模式

            with torch.no_grad():  # 模型不需要梯度更新
                for j, data in enumerate(valid_loader):  # 获取次数索引与验证集数据
                    inputs, labels = data  # 验证集的RGB图像和相应标签
                    inputs, labels = inputs.to(device), labels.to(device)  # 推至运算设备

                    bs, ncrops, c, h, w = inputs.size()
                    outputs = googlenet_model(inputs.view(-1, c, h, w))
                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)

                    loss = criterion(outputs_avg, labels)  # 计算验证loss

                    _, predicted = torch.max(outputs_avg.data, 1)  # 获取top1的值与相应索引
                    total_val += labels.size(0)  # 验证总数
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()  # 验证正确数量

                    loss_val += loss.item()  # 获取验证loss

                loss_val_mean = loss_val/len(valid_loader)  # 计算验证loss平均值
                valid_curve.append(loss_val_mean)  # 加入验证曲线列表
                print(f"Valid:\t Epoch[{epoch:0>3}/{MAX_EPOCH:0>3}] "
                      f"Iteration[{j + 1:0>3}/{len(valid_loader):0>3}] "
                      f"Loss: {loss_val_mean:.4f} "
                      f"Acc:{correct_val / total_val:.2%}")

            googlenet_model.train()  # 将模型转化为训练模式

    train_x = range(len(train_curve))  # 训练曲线x坐标列表
    train_y = train_curve  # 训练曲线y坐标列表

    train_iters = len(train_loader)  # 每轮共多少iter
    valid_x = np.arange(1, len(valid_curve)+1) * train_iters*val_interval  # 由于valid中记录的是epochloss，需要对记录点进行转换到iterations
    valid_y = valid_curve  # 验证曲线y坐标列表

    plt.plot(train_x, train_y, label='Train')  # 绘制训练loss曲线
    plt.plot(valid_x, valid_y, label='Valid')  # 绘制验证loss曲线

    plt.legend(loc='upper right')  # 设置图例位置
    plt.ylabel('loss value')  # 设置y轴标题
    plt.xlabel('Iteration')  # 设置x轴标题
    plt.show()  # 展示总绘制曲线图





