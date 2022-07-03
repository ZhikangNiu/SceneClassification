-*- coding: utf-8 -*-
# @Time    : 2022-03-23 13:12
# @Author  : Zhikang Niu
# @FileName: train.py
# @Software: PyCharm
import io
import os
import torch
import torch.nn as nn
import numpy as np
import random
from config import get_option
from tqdm import tqdm
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, squeezenet1_1, squeezenet1_0
from dataset import JsonDataset
from torch.utils.tensorboard import SummaryWriter
import logging
import warnings
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
TRAIN_PATH = './19_classes/train/'
VAL_PATH = './19_classes/val/'

# 损失函数的设置
criterion = nn.CrossEntropyLoss()
writer = SummaryWriter(LOG_PATH)

def seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_model(model_choice=0):
    if model_choice == 0:
        model = mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(in_features=1024, out_features=CLASS_NUM, bias=True)
    elif model_choice == 1:
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=CLASS_NUM, bias=True)
    elif model_choice == 2:
        model = squeezenet1_0(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, CLASS_NUM, kernel_size=(1, 1), stride=(1, 1))
    elif model_choice == 3:
        model = squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, CLASS_NUM, kernel_size=(1, 1), stride=(1, 1))
    elif model_choice == 4:
        model = ghostnet()
        model.load_state_dict(torch.load('./checkpoint/state_dict_73.98.pth'))
        model.classifier = nn.Linear(1280,CLASS_NUM)
    return model


def train(epoch, model):
    train_process = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_dataset = ImageFolder(TRAIN_PATH, transform=train_process)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8,shuffle=True)

    optimizer = Adam(model.parameters(), lr=1e-6, betas=[0.9, 0.99], eps=1e-08, weight_decay=0.0)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0)
    model.train()
    train_loss = 0
    for image, target in tqdm(train_dataloader):
        image = image.cuda()
        target = target.cuda()
        predict = model(image)
        loss = criterion(predict, target)
        writer.add_scalar("Loss/Train",loss,epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item() * image.size(0)
    train_loss = train_loss / len(train_dataloader.dataset)
    logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))


def val(epoch, model):
    val_process = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        # transforms.Normalize((0.5,),(0.5,))
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_dataset = ImageFolder(VAL_PATH,transform=val_process)
    val_dataloader = DataLoader(val_dataset,batch_size=32,shuffle=False)
    #val_dataset = MyDataset(VAL_FILE_PATH, VAL_PATH, val_process)
    #val_dataloader = DataLoader(val_dataset, batch_size=32, num_workers=8,shuffle=False)

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
            writer.add_scalar('Loss/val',loss,epoch)
            val_loss += loss.item() * data.size(0)
    val_loss = val_loss / len(val_dataloader)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels == pred_labels) / len(pred_labels)
    writer.add_scalar('acc',acc,epoch)
    logger.info('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))


if __name__ == '__main__':
    # 设置随机数种子
    seed(SEED)
    # 加载模型
    model = get_model(model_choice=1).cuda()
    model.load_state_dict(torch.load("./checkpoint/MobileNetv3_large_epoch_58.pth")["model_state_dict"])
    print("-------load model-------")
    # 训练数据
    for epoch in range(EPOCHS):
        train(epoch,model)
        val(epoch, model)
        logger.info('----------------------------------')
        if epoch % 2 == 0 and epoch > 0:
            if GPUS > 1:
                checkpoint = {'model': model.module,
                            'model_state_dict': model.module.state_dict(),
                            #'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'MobileNetv3_large_2_epoch_{}.pth'.format(epoch)))
            else:
                checkpoint = {'model': model,
                            'model_state_dict': model.state_dict(),
                            # 'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch}
                torch.save(checkpoint, os.path.join(save_folder, 'MobileNetv3_large_epoch_{}.pth'.format(epoch)))