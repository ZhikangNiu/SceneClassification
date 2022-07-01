# -*- coding: utf-8 -*-
# @Time    : 2022-06-25 20:09
# @Author  : Zhikang Niu
# @FileName: level2_acc.py
# @Software: PyCharm



import timm
import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致
import torch
import torch.nn as nn
import numpy as np
import random
from config import get_option
from tqdm import tqdm
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
import logging
from sklearn.metrics import classification_report
import warnings
from dataset import JsonDataset
warnings.filterwarnings('ignore')

# 超参数的设置
opt = get_option()

CLASS_NUM = opt.class_num
BATCH_SIZE = opt.batch_size
LOG_PATH = opt.tensorboard_path
EPOCHS = opt.epochs
SEED = opt.seed
IMG_SIZE = opt.img_size
LOG_FILE = opt.log_file
GPUS  = opt.GPUS
save_folder = opt.checkpoint_dir

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(asctime)s: %(levelname)s: [%(filename)s: %(lineno)d]: %(message)s')
file_handler.setFormatter(formatter)

# print to screen
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)

# add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


# 文件路径

#/home/niuzhikang/src/filelist/annotation_json

TRAIN_JSON_PATH = '/home/niuzhikang/src/filelist/annotation_json/train.json'
TRAIN_IMAGE_PATH = '/home/public/datasets/place365/train/data_256'
VAL_JSON_PATH = '/home/niuzhikang/src/filelist/annotation_json/val.json'
VAL_IMAGE_PATH ='/home/public/datasets/place365/val/val_256'

# 损失函数的设置
criterion = nn.CrossEntropyLoss()
#writer = SummaryWriter(LOG_PATH)

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(model_choice=1):
    if model_choice == 0:
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(in_features=1024, out_features=CLASS_NUM, bias=True)
    elif model_choice == 1:
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=CLASS_NUM, bias=True)
    elif model_choice == 2:
        model = timm.create_model("mobilenetv3_large_100",num_classes=CLASS_NUM,pretrained=True)
    return model



def val(model):
    val_process = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_dataset = JsonDataset(VAL_JSON_PATH,VAL_IMAGE_PATH,transform=val_process)
    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in tqdm(val_dataloader):
            data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item() * data.size(0)
    val_loss = val_loss / len(val_dataloader)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    print('Validation Loss: {:.6f}, Accuracy: {:6f}'.format( val_loss, acc))
    print(gt_labels)
    print("--------")
    print(pred_labels)

    print(classification_report(gt_labels,pred_labels))


if __name__ == '__main__':
    # 设置随机数种子
    seed(SEED)
    # 加载模型
    model = get_model(model_choice=1).cuda()
    model.load_state_dict(torch.load("./checkpoint/MobileNetv3_large_epoch_28.pth")["model_state_dict"])
    print("-------load model-------")
    val(model)
    logger.info('----------------------------------')