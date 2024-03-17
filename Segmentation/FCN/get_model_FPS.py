# -*- coding: utf-8 -*-
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from tools.common_tools import get_net


def img_transform(img_l, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_l: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_l)
    return img_t


def process_img(img_path):
    """
    图像预处理
    :param img_path: 输入图像路径
    :return:
    """
    norm_mean = [0.16328026, 0.16328026, 0.16328026]  # 均值
    norm_std = [0.2432042, 0.2432042, 0.2432042]  # 标准差

    inference_transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像从[0,255]范围转换到[0,1]范围(H×W×C)的tensor格式
        transforms.Normalize(norm_mean, norm_std),  # 归一化张量图像。
    ])

    img_l = Image.open(img_path).convert("RGB")  # 按照指定格式读取图像

    # img --> tensor
    img_t = img_transform(img_l, inference_transform)  # transform
    img_t = img_t.to(device)  # 推至运算设备

    return img_t, img_l


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = f"BaseLine_76e_43.7023%.pkl"
    path_state_dict = os.path.join(BASE_DIR, "results", "BaseLine_P100_B8_test", model_name)
    path_fps_log = os.path.join(BASE_DIR, "fps_report")
    fps_log_name = os.path.join(path_fps_log, f"Custom_seg.log")
    path_img = os.path.join(BASE_DIR, "image", '69.jpg')

    if os.path.exists(fps_log_name):
        os.remove(fps_log_name)

    os.makedirs(path_fps_log, exist_ok=True)

    cycle = 10
    cycle_number = 200

    # 加载模型
    CustomNet = get_net(device=device, vis_model=False, path_state_dict=path_state_dict)

    # 加载图像
    img_tensor, img_l = process_img(path_img)

    for i in range(cycle):
        # 预测并计时
        with torch.no_grad():
            time_tic = time.time()
            for _ in range(cycle_number):
                outputs = CustomNet(img_tensor.unsqueeze(0))
            time_toc = time.time()

        # 计算花费时间
        cost_time = time_toc - time_tic
        per_ms = (cost_time / cycle_number) * 1000
        fps = (1000 / per_ms)

        with open(f"{fps_log_name}", "a") as f:
            f.writelines(f"Done {cycle_number} time "
                         f"consuming: {cost_time:.8f}s, "
                         f"fps: {fps:.5} img/s, "
                         f"times per image: {per_ms:.5} ms/img\n")
        f.close()

        print(f"Done {cycle_number} time "
              f"consuming: {cost_time:.8f}s, "
              f"fps: {fps:.5} img/s, "
              f"times per image: {per_ms:.5} ms/img")
