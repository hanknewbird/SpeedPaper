import numpy as np


class runningScore(object):
    def __init__(self, n_classes, ignore_index=None):
        """
        初始化
        :param n_classes: database的类别,包括背景
        :param ignore_index: 需要忽略的类别id,一般为未标注id, eg. CamVid.id_unlabel
        """

        self.n_classes = n_classes  # 类别数
        self.confusion_matrix = np.zeros((n_classes, n_classes))  # N*N的混淆矩阵

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):  # 判断类型是否为int
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def _fast_hist(self, label_true, label_pred, n_class):
        """
        通过真是标签与预测标签,返回一个N*N的混淆矩阵
        :param label_true: 真实标签
        :param label_pred: 预测标签
        :param n_class: 类别数量
        :return: N*N的混淆矩阵
        """

        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        """
        更新混淆矩阵
        :param label_trues: 真实标签
        :param label_preds: 预测标签
        :return: 更新后的混淆矩阵
        """

        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """
        计算各种评估分数
        - pixel_acc:
        - class_acc: class mean acc
        - mIou :     mean intersection over union
        - fwIou:     frequency weighted intersection union
        :return:
        """

        # 混淆矩阵
        hist = self.confusion_matrix

        # 忽略部分标签
        if self.ignore_index is not None:
            for index in self.ignore_index:
                hist = np.delete(hist, index, axis=0)
                hist = np.delete(hist, index, axis=1)

        acc = np.diag(hist).sum() / hist.sum()  # 准确率
        acc_cls = np.diag(hist) / hist.sum(axis=1)  # 类准确率
        acc_cls = np.nanmean(acc_cls)  # 忽略NaN值的平均值 类平均准确率
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))  # 计算iou
        mean_iou = np.nanmean(iu)  # 计算忽略NaN值的数组平均值
        freq = hist.sum(axis=1) / hist.sum()
        fw_iou = (freq[freq > 0] * iu[freq > 0]).sum()  # 加权交并比

        # set unlabel as nan
        if self.ignore_index is not None:
            for index in self.ignore_index:
                iu = np.insert(iu, index, np.nan)

        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "pixel_acc: ": acc,
                "class_acc: ": acc_cls,
                "mIou: ": mean_iou,
                "fwIou: ": fw_iou,
            },
            cls_iu,
        )

    def reset(self):
        """
        重置混淆矩阵
        """
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """
    计算、存储平均值与当前值
    """

    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
