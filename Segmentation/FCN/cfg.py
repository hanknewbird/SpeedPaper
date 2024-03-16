import os

BATCH_SIZE = 2  # 批大小
EPOCH_NUMBER = 2  # 训练轮数
DATASET = ['CamVid', 12]  # 数据集文件夹名称与类别数

crop_size = (352, 480)  # 裁剪尺寸

data = "Data"
class_dict_path = os.path.join("..", data, DATASET[0], 'class_dict.csv')  # 涂色路径

# 训练集
TRAIN_ROOT = os.path.join("..", data, DATASET[0], 'train')
TRAIN_LABEL = os.path.join("..", data, DATASET[0], 'train_labels')

# 验证集
VAL_ROOT = os.path.join("..", data, DATASET[0], 'val')
VAL_LABEL = os.path.join("..", data, DATASET[0], 'val_labels')

# 测试集
TEST_ROOT = os.path.join("..", data, DATASET[0], 'test')
TEST_LABEL = os.path.join("..", data, DATASET[0], 'test_labels')

OUTPUT_PIC_ROOT = os.path.join("Results", "result_pics")
OUTPUT_MODEL_ROOT = os.path.join("Results", "weights")
