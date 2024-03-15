# -*- coding: utf-8 -*-
import os
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tools.cifar10_dataset import CifarDataset
from tools.resnet import resnet56, resnet20
from tools.common_tools import ModelTrainer, show_confMat, plot_line

BASE_DIR = os.path.dirname(os.path.abspath(__file__))                  # 路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备

if __name__ == "__main__":

    # 数据集地址
    train_dir = os.path.join(BASE_DIR, "..", "Data", "cifar-10", "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "Data", "cifar-10", "cifar10_test")

    start_time = 0
    end_time = 0

    now_time = datetime.now()                                    # 时间
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')        # 时间格式化
    log_dir = os.path.join(BASE_DIR, "..", "result", time_str)  # 结果保存路径

    os.makedirs(log_dir, exist_ok=True)

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # cifar10的类名

    MAX_EPOCH = 182         # 最大训练轮数    64000 / (45000 / 128) = 182 epochs
    BATCH_SIZE = 8          # 每次处理多少图片
    LR = 0.1                # 初始学习率
    start_epoch = -1        # epoch记录参数
    milestones = [92, 136]  # 学习率在第多少个epoch下降10倍

    # ============================ step 1/5 数据 ============================

    norm_mean = [0.485, 0.456, 0.406]  # 数据集均值
    norm_std = [0.229, 0.224, 0.225]   # 数据集方差

    train_transform = transforms.Compose([          # 训练集的transform
        transforms.Resize(32),                      # 将图片处理为32*32
        transforms.RandomHorizontalFlip(p=0.5),     # 随机翻转
        transforms.RandomCrop(32, padding=4),   # 随机裁剪
        transforms.ToTensor(),                      # 将图像转化为tensor格式,输出为[0,1]的tensor
        transforms.Normalize(norm_mean, norm_std),  # 归一化,便于模型拟合
    ])

    valid_transform = transforms.Compose([          # 训练集的transform
        transforms.Resize((32, 32)),                # 将图片处理为32*32
        transforms.ToTensor(),                      # 将图像转化为tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化,便于模型拟合
    ])

    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CifarDataset(data_dir=test_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # 使用2个子进程加载数据
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE, num_workers=2)                # 使用2个子进程加载数据

    # ============================ step 2/5 模型 ============================

    resnet_model = resnet56()  # 定义模型结构

    resnet_model.to(device)    # 将模型推至运算设备

    # ============================ step 3/5 损失函数 ============================

    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数计算loss

    # ============================ step 4/5 优化器 ============================

    # 冻结卷积层
    optimizer = optim.SGD(resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # 优化器

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)   # 学习率下降策略：下降10倍

    # ============================ step 5/5 训练 ============================
    loss_rec = {"train": [], "valid": []}  # Loss曲线数据
    acc_rec = {"train": [], "valid": []}   # Accuracy曲线数据
    best_acc, best_epoch = 0, 0            # 记录最好的Accuracy和相应的epoch

    start_time = datetime.now()            # 训练开始时间

    for epoch in range(start_epoch + 1, MAX_EPOCH):  # 模型开始训练

        # 得到平均loss,平均accuracy和混淆矩阵
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, resnet_model, criterion, device)

        print(f"Epoch[{epoch + 1:0>3}/{MAX_EPOCH:0>3}] "
              f"Train Acc: {acc_train:.2%} "
              f"Valid Acc:{acc_valid:.2%} "
              f"Train loss:{loss_train:.4f} "
              f"Valid loss:{loss_valid:.4f} "
              f"LR:{optimizer.param_groups[0]['lr']}")

        scheduler.step()  # 更新学习率

        # 补全loss和Accuracy曲线
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        # 绘制混淆矩阵
        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH - 1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH - 1)

        plt_x = np.arange(1, epoch + 2)  # 计算x轴
        # 绘制loss和Accuracy曲线
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        # 判断并保存模型
        if epoch > (MAX_EPOCH / 2) and best_acc < acc_valid:  # 若验证Accuracy大于原先最好的Accuracy,且模型轮数大于1/2,则保存模型
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": resnet_model.state_dict(),    # 模型参数
                          "optimizer_state_dict": optimizer.state_dict(),   # 优化器
                          "epoch": epoch,                                   # best_epoch
                          "best_acc": best_acc}                             # best_accuracy

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")  # 模型保存路径
            torch.save(checkpoint, path_checkpoint)                         # 保存模型

    end_time = datetime.now()                                               # 训练结束时间

    spend_time = (end_time - start_time).seconds
    print(f"Model training completed, "
          f"start time: {datetime.strftime(start_time.now(), '%m-%d_%H-%M')}, "
          f"end time: {datetime.strftime(end_time, '%m-%d_%H-%M')}, "
          f"spend time: {spend_time}s "
          f"best acc: {best_acc} in :{best_epoch} epochs.")
