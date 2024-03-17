import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 基础路径
DATA_DIR = os.path.join(BASE_DIR, "..", "Data", "DAGM")  # 数据集路径
DATA_MEAN = [0.16328026, 0.16328026, 0.16328026]  # 均值
DATA_STD = [0.2432042, 0.2432042, 0.2432042]  # 标准差

# MODEL_PATH = os.path.join(BASE_DIR, "model", "CheapNet_G64.pkl")  # 预训练模型路径
MODEL_PATH = None
SAVE_MODEL = True

MAX_EPOCH = 100  # 跑多少轮
BATCH_SIZE = 8  # 每次载入多少图片
DATALOADER_WORKERS = 8  # dataloader线程数

TIME_STR = datetime.strftime(datetime.now(), '%m-%d-%H-%M')  # 时间格式化

LR = 0.01  # 学习率
MILESTONES = [50, 80]  # 学习率在第多少个epoch下降
GAMMA = 0.1  # 下降参数

TAG = "BaseLine"  # 备注
LOG_DIR = os.path.join(BASE_DIR, "results", f"{TAG}_P{MAX_EPOCH}_B{BATCH_SIZE}_{TIME_STR}")  # 结果保存路径
log_name = f'{TIME_STR}.log'
SAVE_MODEL_NUM = 3  # 保存效果最好的模型数量
CONTRAST_IMG_PATH = os.path.join(BASE_DIR, "image", f"69.jpg")  # 对比图像路径
OUT_MODEL = 2  # 1:单张图，2:两张图
VIDEO_FPS = 5
