# -*- coding: utf-8 -*-
# @Time    : 2022-03-23 21:08
# @Author  : Zhikang Niu
# @FileName: dataset.py
# @Software: PyCharm

import imp
from typing import List
import torch
import pandas as pd
import cv2
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np

class TxtDataset(Dataset):
    """
    使用的是Place365原始数据 txt读取方式
    file_path: 读取标注文件的位置
    img_path : 图片数据集的位置
    ignore_labels:传入忽略的类标
    mode : 读取数据集是train还是val
    transform: 对文件的变换操作
    """
    def __init__(self,file_path:str,
                      img_path:str,
                      ignore_labels:List,
                      mode = 'train',
                      transform=None):
        # 读取文件
        file = open(file_path,'r')
        # 传入文件路径参数
        self.img_path = img_path
        label_list = []
        image_list = []
        self.ignore_labels = ignore_labels
        if mode != 'train':
            for i in file:
                label = i.split()[-1]
                image = i.split()[0]
                if int(label) not in ignore_labels:
                    label_list.append(label)
                    image_list.append(image)
        else:
            for i in file:
                label = i.split()[-1]
                # train数据集的路径需要从/以后开始
                image = i.split()[0][1:]
                if int(label) not in ignore_labels:
                    label_list.append(label)
                    image_list.append(image)

        self.images = image_list
        self.labels = label_list
        self.transform = transform


    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        img_name = self.images[item]
        img_path = os.path.join(self.img_path,img_name)
        # 使用opencv读取数据集，尽量使用cv2-->Image
        img = cv2.imread(img_path)
        #img = Image.open(img_path)
        label = int(self.labels[item])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform =transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        #print(img_path)
        return img,label


class JsonDataset(Dataset):
    """
    使用的是Place365 json读取方式
    file_path: 读取标注文件的位置
    img_path : 图片数据集的位置
    transform: 对文件的变换操作
    """
    def __init__(self,file_path:str,
                      img_path:str,
                      transform=None):
        # 读取文件
        self.df = pd.read_json(file_path)
        # 传入图片路径
        self.img_path = img_path
        self.transform = transform


    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        img_name = self.df['image_dict'][item]['img_path']
        img_path = os.path.join(self.img_path,img_name)
        img = cv2.imread(img_path)
        img = Image.fromarray(np.uint8(img))
        #img = Image.open(img_path)
        label = int(self.df['image_dict'][item]['level_2'])
        if self.transform is not None:
            img = self.transform(img)
        else:
            transform =transforms.Compose([transforms.ToTensor()])
            img = transform(img)
        label = torch.tensor(label)
        return img,label


