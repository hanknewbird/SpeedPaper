# -*- coding: utf-8 -*-
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import torchvision.models as models


def get_resnet_18(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.resnet18()                                # 获取模型结构
    if path_state_dict:                                      # 判断是否传入了预训练模型
        pretrained_state_dict = torch.load(path_state_dict)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict)         # 将预训练模型载入模型
    model.eval()                                             # 模型开启验证状态

    if vis_model:  # 是否打印模型
        from torchsummary import summary
        summary(model, input_size=(3, 32, 32), device="cpu")

    model.to(device)
    return model


def get_resnet_50(path_state_dict, device, vis_model=False):
    """
    创建模型，加载参数
    :param path_state_dict:
    :return:
    """
    model = models.resnet50()
    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict)
        model.load_state_dict(pretrained_state_dict)
    model.eval()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    model.to(device)
    return model


class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch):
        """
        每次传入一个epoch的数据进行模型训练
        :param data_loader: 训练集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param optimizer: 优化器
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :return: 平均loss,平均accuracy和混淆矩阵
        """
        model.train()  # 开启模型训练模式

        conf_mat = np.zeros((10, 10))  # 初始化混淆矩阵
        loss_sigma = []                # 全程loss记录

        for i, data in enumerate(data_loader):  # 迭代训练集加载器,得到iteration和相关图像data

            inputs, labels = data                                  # 通过data得到图像数据和对应的label
            inputs, labels = inputs.to(device), labels.to(device)  # 传入运算设备

            outputs = model(inputs)         # 载入模型,得到预测值

            optimizer.zero_grad()           # 优化器梯度归零
            loss = loss_f(outputs, labels)  # 计算损失
            loss.backward()                 # 反向传播,计算梯度
            optimizer.step()                # 更新梯度

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)  # 得到top1的值和对应的label

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实值
                pre_i = predicted[j].cpu().numpy()  # 预测值
                conf_mat[cate_i, pre_i] += 1.       # 一步一步填充混淆矩阵

            # 统计loss
            loss_sigma.append(loss.item())               # 记录每次的loss值
            acc_avg = conf_mat.trace() / conf_mat.sum()  # Accuracy = 混淆矩阵对角线总和/混淆矩阵总和

            # 每50个iteration打印一次训练信息
            if i % 50 == 50 - 1:
                print(f"Training: "
                      f"Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] "
                      f"Iteration[{i + 1:0>3}/{len(data_loader):0>3}] "
                      f"Loss: {np.mean(loss_sigma):.4f} "
                      f"Acc:{acc_avg:.2%}")

        return np.mean(loss_sigma), acc_avg, conf_mat

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        """
        模型验证
        :param data_loader: 验证集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param device: 运算设备
        :return: 平均loss,平均accuracy和混淆矩阵
        """
        model.eval()  # 模型验证模式

        conf_mat = np.zeros((10, 10))  # 混淆矩阵
        loss_sigma = []                # 全程loss记录

        for i, data in enumerate(data_loader):  # 迭代验证集加载器,得到iteration和相关图像data

            inputs, labels = data                                  # 通过data得到图像数据和对应的label
            inputs, labels = inputs.to(device), labels.to(device)  # 传入运算设备

            outputs = model(inputs)  # 载入模型,得到预测值

            loss = loss_f(outputs, labels)  # 计算loss

            # 统计预测信息
            _, predicted = torch.max(outputs.data, 1)  # 得到top1的值和对应的label

            # 统计混淆矩阵
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()    # 真实值
                pre_i = predicted[j].cpu().numpy()  # 预测值
                conf_mat[cate_i, pre_i] += 1.       # 一步一步填充混淆矩阵

            # 统计loss
            loss_sigma.append(loss.item())  # 记录每次的loss值

        acc_avg = conf_mat.trace() / conf_mat.sum()  # Accuracy = 混淆矩阵对角线总和/混淆矩阵总和

        return np.mean(loss_sigma), acc_avg, conf_mat


def show_confMat(confusion_mat, classes, set_name, out_dir, verbose=False):
    """
    混淆矩阵图绘制
    :param confusion_mat: 混淆矩阵
    :param classes: 类别名
    :param set_name: train/valid
    :param out_dir: 保存文件夹路径
    :return:
    """
    cls_num = len(classes)  # 类别名

    # 归一化
    confusion_mat_N = confusion_mat.copy()                                       # 复制一份混淆矩阵
    for i in range(len(classes)):                                                # 循环每个类别
        confusion_mat_N[i, :] = confusion_mat[i, :] / confusion_mat[i, :].sum()  # 每行循环归一化

    # 获取颜色
    cmap = plt.cm.get_cmap('Greys')         # 更多颜色: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.imshow(confusion_mat_N, cmap=cmap)  # 使用matplotlib接收归一化后的混淆矩阵值,并设置相应的颜色贴图
    plt.colorbar()                          # 向绘图添加色条

    # 设置文字
    xlocations = np.array(range(len(classes)))          # 将类目对应到相关x坐标
    plt.xticks(xlocations, list(classes), rotation=60)  # 设置旋转60°后的x刻度
    plt.yticks(xlocations, list(classes))               # 设置y刻度
    plt.xlabel('Predict label')                         # 设置x轴标签
    plt.ylabel('True label')                            # 设置y轴标签
    plt.title('Confusion_Matrix_' + set_name)           # 设置title

    # 打印数字
    for i in range(confusion_mat_N.shape[0]):      # 每行
        for j in range(confusion_mat_N.shape[1]):  # 每列
            plt.text(x=j, y=i, s=int(confusion_mat[i, j]), va='center', ha='center', color='red', fontsize=10)

    plt.savefig(os.path.join(out_dir, 'Confusion_Matrix_' + set_name + '.png'))  # 保存
    plt.show()                                                                  # 显示

    if verbose:
        for i in range(cls_num):
            print(f'class:{classes[i]:<10}, '
                  f'total num:{np.sum(confusion_mat[i, :]):<6}, '
                  f'correct num:{confusion_mat[i, i]:<5}  '
                  f'Recall: {confusion_mat[i, i] / (.1 + np.sum(confusion_mat[i, :])):.2%} '
                  f'Precision: {confusion_mat[i, i] / (.1 + np.sum(confusion_mat[:, i])):.2%}')


def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir: 保存路径
    :return:
    """
    # 绘制线段
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))  # 设置y轴标签
    plt.xlabel('Epoch')    # 设置x轴标签

    location = 'upper right' if mode == 'loss' else 'upper left'  # 设置图例位置
    plt.legend(loc=location)                                      # 图例设置

    plt.title('_'.join([mode]))                        # 设置title
    plt.savefig(os.path.join(out_dir, mode + '.png'))  # 保存
    plt.show()                                         # 展示
