import numpy as np


class RunningScore(object):
    def __init__(self, n_classes, ignore_index=None):
        """
        初始化
        :param n_classes: database的类别,包括背景
        :param ignore_index: 需要忽略的类别id,一般为未标注id
        """

        self.n_classes = n_classes  # 类别数
        self.confusion_matrix = np.zeros((n_classes, n_classes))  # 创建N*N的空混淆矩阵
        self.cls_name = ["背景", "折皱", "擦伤", "脏污", "针孔"]

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):  # 判断类型是否为int
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred):
        """
        通过真实标签与预测标签,返回一个N*N的混淆矩阵
        :param label_true: 真实标签
        :param label_pred: 预测标签
        :return: N*N的混淆矩阵
        """
        # 将整数掩码转换为整数类型
        label_true = label_true.astype(int)

        # 将标签转换为混淆矩阵的索引
        index = self.n_classes * label_true + label_pred

        # 使用np.bincount计算混淆矩阵,用于计算数组中每个非负整数值出现的次数
        hist = np.bincount(
            index, minlength=self.n_classes ** 2
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        """
        更新混淆矩阵
        :param label_trues: 真实标签
        :param label_preds: 预测标签
        :return: 更新后的混淆矩阵
        """

        for lt, lp in zip(label_trues, label_preds):  # 迭代每张标签和预测图
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_scores(self):
        """
        计算各种评估分数
        - pixel_acc: 像素
        - class_acc: class mean acc
        - mIou :     mean intersection over union
        - fwIou:     frequency weighted intersection union
        :return:
        """

        # 获取全局混淆矩阵
        hist = self.confusion_matrix

        # 忽略部分标签
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc_all, acc_cls = self.get_accuracy(hist)  # 整体准确率, 类准确率
        MIoU, IoU_cls = self.get_jaccard_index(hist)
        dice_all, dice_cls = self.get_dice(hist)
        precision_all, precision_cls, recall_all, recall_cls = self.get_precision_recall(hist)

        # 设置unlabel为Nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                MIoU = np.insert(MIoU, index, np.nan)

        return (
            {
                "all_acc": acc_all,
                "all_mIou": MIoU,
                "all_dice": dice_all,
                "all_precision": precision_all,
                "all_recall": recall_all,
            },
            {
                "class_acc": acc_cls,
                "class_iou": IoU_cls,
                "class_dice": dice_cls,
                "class_precision": precision_cls,
                "class_recall": recall_cls,

            },
            self.confusion_matrix
        )

    def reset(self):
        """重置混淆矩阵"""
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def get_accuracy(self, confusion_matrix):
        # 将混淆矩阵的对角线元素相加来获取分类正确的样本数，然后除以总样本数得到准确率。
        acc_all = np.diag(confusion_matrix).sum() / confusion_matrix.sum()  # 整体准确率

        # 计算每个类别的准确率，即对角线上的值除以每个类别的样本总数。
        acc_cls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)  # 类准确率

        # 将类别名与对应的 acc 组合成字典
        acc_cls = dict(zip(self.cls_name, acc_cls))

        return acc_all, acc_cls

    def get_jaccard_index(self, confusion_matrix):
        # 计算每个类别的真正例和假正例
        true_positive = np.diag(confusion_matrix)
        false_positive = confusion_matrix.sum(axis=0) - true_positive
        false_negative = confusion_matrix.sum(axis=1) - true_positive

        # 计算每个类别的 IoU
        intersection = true_positive
        union = true_positive + false_positive + false_negative
        iou_cls = intersection / union

        # 计算整体IoU
        iou_all = np.nanmean(iou_cls)

        # 将类别名与对应的 IoU 组合成字典
        iou_cls = dict(zip(self.cls_name, iou_cls))

        return iou_all, iou_cls

    def get_dice(self, confusion_matrix):
        # 计算每个类别的真阳性和假阳性
        true_positive = np.diag(confusion_matrix)
        false_positive = confusion_matrix.sum(axis=0) - true_positive
        false_negative = confusion_matrix.sum(axis=1) - true_positive

        # 计算Dice
        dice_cls = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)

        # 忽略NaN值并计算类别Dice的平均值
        dice_all = np.nanmean(dice_cls)

        dice_cls = dict(zip(self.cls_name, dice_cls))

        return dice_all, dice_cls

    def get_precision_recall(self, confusion_matrix):
        # 计算每个类别的真阳性、假阳性和假阴性
        true_positive = np.diag(confusion_matrix)
        false_positive = confusion_matrix.sum(axis=0) - true_positive
        false_negative = confusion_matrix.sum(axis=1) - true_positive

        # 计算Precision 和 Recall
        precision_cls = true_positive / (true_positive + false_positive)
        recall_cls = true_positive / (true_positive + false_negative)

        # 忽略NaN值并计算类别Precision 和 Recall 的平均值
        precision_all = np.nanmean(precision_cls)
        recall_all = np.nanmean(recall_cls)

        precision_cls = dict(zip(self.cls_name, precision_cls))
        recall_cls = dict(zip(self.cls_name, recall_cls))

        return precision_all, precision_cls, recall_all, recall_cls
