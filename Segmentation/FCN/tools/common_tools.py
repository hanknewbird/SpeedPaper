# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torchvision
from PIL import Image
from .net.FCN import FCN
import torch
import os
import logging
import sys
from .metrics import RunningScore


def get_net(device, vis_model=False, path_state_dict=None):
    """
    创建模型，加载参数
    :param device: 运算设备
    :param vis_model: 是否打印模型结构
    :param path_state_dict:
    :return: 预训练模型
    """
    model = FCN(5)  # 创建模型结构

    if path_state_dict:
        pretrained_state_dict = torch.load(path_state_dict, map_location=device)  # 读取预训练模型
        model.load_state_dict(pretrained_state_dict['CustomNet'])  # 将预训练模型载入模型

    model.eval()  # 开启验证模式

    if vis_model:  # 是否打印模型结构
        from torchinfo import summary
        summary(model, input_size=(1, 3, 640, 480), device="cpu")

    model.to(device)  # 将模型推至运算设备
    return model


class CustomNetTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch, logger):
        """
        每次传入一个epoch的数据进行模型训练
        :param data_loader: 训练集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param optimizer: 优化器
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.train()  # 开启模型训练模式

        loss_avg = []  # 平均loss
        for i, data in enumerate(data_loader):  # 迭代训练集加载器,得到iteration和相关图像data

            x, target = data  # 通过data得到图像数据
            x = x.to(device)  # 传入运算设备
            target = target.squeeze(1).long()
            target = target.to(device)  # 传入运算设备

            y = model(x)  # 载入模型,得到预测值

            optimizer.zero_grad()  # 优化器梯度归零
            loss = loss_f(y, target)  # 计算每个预测值与target的损失
            loss.backward()  # 反向传播,计算梯度
            optimizer.step()  # 更新梯度

            loss_avg.append(loss.item())  # 记录每次的loss值

            logger.info(f'Train | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                        f'Train loss: {np.mean(loss_avg):.8f}')

        return np.mean(loss_avg)

    @staticmethod
    def valid(data_loader, model, loss_f, epoch_id, device, max_epoch, logger):
        """
        模型验证
        :param data_loader: 验证集加载器
        :param model: 模型
        :param loss_f: 损失函数
        :param epoch_id: 第几个epoch
        :param device: 运算设备
        :param max_epoch: 最大训练轮数
        :param logger: 日志
        :return: 平均loss
        """
        model.eval()  # 模型验证模式
        running_metrics_val = RunningScore(5)  # 创建5*5的混淆矩阵

        loss_avg = []  # 平均loss

        for i, data in enumerate(data_loader):  # 迭代验证集加载器,得到iteration和相关data
            running_metrics_val.reset()  # 初始化混淆矩阵

            x, target = data  # 通过data得到图像数据和对应的label
            x = x.to(device)  # 传入运算设备
            target = target.squeeze(1).long()
            target = target.to(device)  # 传入运算设备

            y = model(x)  # 载入模型,得到预测值

            loss = loss_f(y, target)  # 计算loss
            loss_avg.append(loss.item())  # 记录每次的loss值

            logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] '
                        f'Iteration[{i + 1:0>3}/{len(data_loader):0>3}] '
                        f'Valid loss: {np.mean(loss_avg):.4f}')

            predict = y.max(dim=1)[1].data.cpu().numpy()
            label = target.data.cpu().numpy()
            running_metrics_val.update(label, predict)  # 更新混淆矩阵

        metrics = running_metrics_val.get_scores()
        valid_miou = metrics[0]['all_mIou']
        valid_acc = metrics[0]['all_acc']
        valid_dice = metrics[0]['all_dice']
        valid_precision = metrics[0]['all_precision']
        valid_recall = metrics[0]['all_recall']

        valid_class_iu = metrics[1]['class_iou']
        valid_class_acc = metrics[1]['class_acc']
        valid_class_dice = metrics[1]['class_dice']
        valid_class_precision = metrics[1]['class_precision']
        valid_class_recall = metrics[1]['class_recall']
        valid_confusion_matrix = metrics[2]

        logger.info("============================================================================")
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] MIou: {valid_miou}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Accuracy: {valid_acc}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Dice: {valid_dice}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Precision: {valid_precision}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Recall: {valid_recall}')
        logger.info("============================================================================")
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Class IoU: {valid_class_iu}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Class Accuracy: {valid_class_acc}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Class Dice: {valid_class_dice}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Class Precision: {valid_class_precision}')
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] Class Recall: {valid_class_recall}')
        logger.info("============================================================================")
        np.set_printoptions(suppress=True)  # 设置浮点数打印格式
        logger.info(f'Valid | Epoch[{epoch_id + 1:0>3}/{max_epoch:0>3}] confusion_matrix: \n{valid_confusion_matrix}')
        logger.info("============================================================================")

        return np.mean(loss_avg), valid_miou

    @staticmethod
    def get_model_output(model, image_path, device, valid_transform, output_model, epoch, output_dir):

        model.eval()  # 模型验证模式

        img = Image.open(image_path).convert("RGB")  # RGB图
        img_tensor = valid_transform(img).to(device)

        # 预测
        with torch.no_grad():
            outputs = model(img_tensor.unsqueeze(0))
        mask = outputs.max(dim=1)[1].data.cpu().numpy().squeeze(0)

        # 定义每个类别的颜色映射
        class_colors = {
            0: [0, 0, 0],  # 背景，黑色
            1: [255, 0, 0],  # 折皱，红色
            2: [0, 255, 0],  # 擦伤，绿色
            3: [0, 0, 255],  # 脏污，蓝色
            4: [255, 255, 0]  # 针孔，黄色
        }

        # 创建一个空白的彩色图像
        h, w = mask.shape
        colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # 为每个像素赋予颜色
        for i in range(h):
            for j in range(w):
                colored_mask[i, j] = class_colors[mask[i, j]]

        pre_img_torch = torch.from_numpy(colored_mask)

        if output_model == 2:
            img_ori_torch = torch.from_numpy(np.array(img)).transpose(0, 2).transpose(1, 2).unsqueeze(
                0)  # (1, 3,480,640)
            pre_img_torch = pre_img_torch.transpose(0, 2).transpose(1, 2).unsqueeze(0)  # (1, 3,480,640)

            output_img = torch.cat([img_ori_torch, pre_img_torch], dim=0)  # (2, 3,480,640)
            output_img = torchvision.utils.make_grid(output_img, nrow=2, padding=10)  # (3, 500, 1310) chw
        else:
            output_img = pre_img_torch

        # 保存推理图片
        effect_dir = os.path.join(output_dir, "effect")
        save_tensor_images(output_img, effect_dir, epoch)

        return output_img


def save_tensor_images(output_img_tensor, output_folder, epoch):
    # 将Tensor转换为NumPy数组
    output_img_np = output_img_tensor.numpy().transpose(1, 2, 0)  # 2:(3, 500, 1310)==>(500, 1310, 3)
    output_img_np = output_img_np.astype(np.uint8)

    # 创建 OpenCV 的 Mat 对象
    output_img_np = cv2.cvtColor(output_img_np, cv2.COLOR_RGB2BGR)  # 如果 output_img_np 是 RGB 格式的图像数组

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 右上角数字
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size, _ = cv2.getTextSize(str(epoch), font, font_scale, font_thickness)
    text_x = output_img_np.shape[1] - text_size[0] - 10  # 文本的 x 坐标
    text_y = text_size[1] + 10  # 文本的 y 坐标
    cv2.putText(output_img_np, str(epoch), (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness,
                cv2.LINE_AA)

    # 保存图像
    img_path = os.path.join(output_folder, f"epoch_{epoch}.png")
    cv2.imwrite(img_path, output_img_np)


def images_to_video(image_folder, fps):
    effect_dir = os.path.join(image_folder, "effect")
    images = [img for img in os.listdir(effect_dir) if img.endswith(".png")]
    images.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))

    frame = cv2.imread(os.path.join(effect_dir, images[0]))
    height, width, _ = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video_path = os.path.join(image_folder, "video")

    os.makedirs(output_video_path, exist_ok=True)
    output_video_path = os.path.join(output_video_path, 'TrainingProcess.mp4')
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(effect_dir, image)))

    cv2.destroyAllWindows()
    video.release()


def get_logger(log_dir, log_name):
    log_file = os.path.join(log_dir, log_name)

    # 创建log
    logger = logging.getLogger('train')  # log初始化
    logger.setLevel(logging.INFO)  # 设置log级别, INFO是程序正常运行时输出的信息

    # Formatter 设置日志输出格式
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler 日志输出1 -> 输出到控制台
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler 日志输出2 -> 保存到文件
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger
