import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime
from dataset import LoadDataset
from Models import FCN
import cfg
from metrics import *


device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
Load_val = LoadDataset([cfg.VAL_ROOT, cfg.VAL_LABEL], cfg.crop_size)

train_data = DataLoader(Load_train, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)
val_data = DataLoader(Load_val, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=1)


fcn = FCN.FCN(num_class)
fcn = fcn.to(device)
criterion = nn.NLLLoss().to(device)
optimizer = optim.Adam(fcn.parameters(), lr=1e-4)


def train(model):
    best = [0]
    train_loss = 0
    net = model.train()
    running_metrics_val = runningScore(12)

    # 训练轮次
    for epoch in range(cfg.EPOCH_NUMBER):
        running_metrics_val.reset()
        print('Epoch is [{}/{}]'.format(epoch + 1, cfg.EPOCH_NUMBER))
        if epoch % 50 == 0 and epoch != 0:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        # 训练批次
        for i, sample in enumerate(train_data):
            # 载入数据
            img_data = Variable(sample['img'].to(device))
            img_label = Variable(sample['label'].to(device))
            # 训练
            out = net(img_data)
            out = F.log_softmax(out, dim=1)
            loss = criterion(out, img_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            true_label = img_label.data.cpu().numpy()
            running_metrics_val.update(true_label, pre_label)

        metrics = running_metrics_val.get_scores()
        for k, v in metrics[0].items():
            print(k, v)
        train_miou = metrics[0]['mIou: ']
        if max(best) <= train_miou:
            best.append(train_miou)
            t.save(net.state_dict(), './Results/weights/FCN_weight/{}.pth'.format(epoch))


def evaluate(model):
    net = model.eval()
    running_metrics_val = runningScore(12)
    eval_loss = 0
    prec_time = datetime.now()

    for j, sample in enumerate(val_data):
        valImg = Variable(sample['img'].to(device))
        valLabel = Variable(sample['label'].long().to(device))

        out = net(valImg)
        out = F.log_softmax(out, dim=1)
        loss = criterion(out, valLabel)
        eval_loss = loss.item() + eval_loss
        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        true_label = valLabel.data.cpu().numpy()
        running_metrics_val.update(true_label, pre_label)
    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)

    cur_time = datetime.now()
    h, remainder = divmod((cur_time - prec_time).seconds, 3600)
    m, s = divmod(remainder, 60)
    time_str = 'Time: {:.0f}:{:.0f}:{:.0f}'.format(h, m, s)
    print(time_str)


if __name__ == "__main__":
    train(fcn)
