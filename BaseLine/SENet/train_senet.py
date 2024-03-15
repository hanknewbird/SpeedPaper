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
from tools.common_tools import ModelTrainer, show_confMat, plot_line
from tools.se_resnet import CifarSEBasicBlock
from tools.resnet import resnet20

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 运算设备


if __name__ == "__main__":

    # 参数配置  如果缺少数据集或预训练文件，请阅读相关readme获取下载地址
    # 数据集："https://github.com/hanknewbird/SpeedPaper/tree/main/BaseLine/Data"
    # 预训练文件："https://github.com/hanknewbird/SpeedPaper/tree/main/BaseLine/ModelFile"
    train_dir = os.path.join(BASE_DIR, "..", "Data", "cifar-10",  "cifar10_train")
    test_dir = os.path.join(BASE_DIR, "..", "Data", "cifar-10", "cifar10_test")

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    log_dir = os.path.join(BASE_DIR, "..", "result", time_str)
    os.makedirs(log_dir, exist_ok=True)

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    num_classes = 10         # 定义类别
    MAX_EPOCH = 3            # 跑多少轮
    BATCH_SIZE = 8           # 每次载入多少图片
    LR = 0.1                 # 学习率
    log_interval = 1         # 多少次打印一次log
    val_interval = 1         # 多少次验证一次
    start_epoch = -1         # 开始的epoch(断点训练时有用)
    milestones = [150, 225]  # 学习率衰减步数

    # ============================ step 1/5 数据 ============================

    norm_mean = [0.485, 0.456, 0.406]  # imagenet上的均值
    norm_std = [0.229, 0.224, 0.225]  # imagenet上的方差

    train_transform = transforms.Compose([          # 定义图片的预训练方式
        transforms.Resize(32),                      # (32)是短边压缩，注意与(32, 32)的区别
        transforms.RandomHorizontalFlip(p=0.5),     # 随机水平翻转
        transforms.RandomCrop(32, padding=4),   # 随机裁剪为(224,224)
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((32, 32)),                # 裁剪为(32,32)
        transforms.ToTensor(),                      # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像
    ])

    # 构建MyDataset实例
    train_data = CifarDataset(data_dir=train_dir, transform=train_transform)
    valid_data = CifarDataset(data_dir=test_dir, transform=valid_transform)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_loader = DataLoader(dataset=valid_data, batch_size=32, num_workers=2)

    # ============================ step 2/5 模型 ============================

    se_resnet_model = resnet20()
    se_resnet_model.to(device)  # 推至运算设备上

    # ============================ step 3/5 损失函数 ============================

    criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

    # ============================ step 4/5 优化器 ============================

    # 冻结卷积层
    optimizer = optim.SGD(se_resnet_model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)  # 选择优化器
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)

    # ============================ step 5/5 训练 ============================

    loss_rec = {"train": [], "valid": []}
    acc_rec = {"train": [], "valid": []}
    best_acc, best_epoch = 0, 0

    for epoch in range(start_epoch + 1, MAX_EPOCH):

        # 训练(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch)
        loss_train, acc_train, mat_train = ModelTrainer.train(train_loader, se_resnet_model, criterion, optimizer, epoch, device, MAX_EPOCH)
        loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, se_resnet_model, criterion, device)
        print(f"Epoch[{epoch + 1:0>3}/{MAX_EPOCH:0>3}] "
              f"Train Acc: {acc_train:.2%} "
              f"Valid Acc:{acc_valid:.2%} "
              f"Train loss:{loss_train:.4f} "
              f"Valid loss:{loss_valid:.4f} LR:{optimizer.param_groups[0]['lr']}")

        scheduler.step()  # 更新学习率

        # 绘图
        loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["train"].append(acc_train), acc_rec["valid"].append(acc_valid)

        show_confMat(mat_train, class_names, "train", log_dir, verbose=epoch == MAX_EPOCH-1)
        show_confMat(mat_valid, class_names, "valid", log_dir, verbose=epoch == MAX_EPOCH-1)

        plt_x = np.arange(1, epoch+2)
        plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["train"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if epoch > (MAX_EPOCH/2) and best_acc < acc_valid:
            best_acc = acc_valid
            best_epoch = epoch

            checkpoint = {"model_state_dict": se_resnet_model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}

            path_checkpoint = os.path.join(log_dir, "checkpoint_best.pkl")
            torch.save(checkpoint, path_checkpoint)

    print(f"done!!!{datetime.strftime(datetime.now(), '%m-%d_%H-%M')}, "
          f"best acc: {best_acc} in :{best_epoch} epochs. ")
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
    print(time_str)
