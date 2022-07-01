# -*- coding: utf-8 -*-
# @Time    : 2022-02-24 23:49
# @Author  : Zhikang Niu
# @FileName: resnet_load.py
# @Software: PyCharm
import os
import cv2
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import cv2


class Predicter():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        # self.CLASS_NUM = 80

    def set_seed(self, seed=66):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def img_load(self, img_path):
        # TODO: 看下opencv奇奇怪怪的问题
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path).convert("RGB")
        transformer = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=1),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        img_tensor = transformer(img)
        x = img_tensor.unsqueeze(0).to(self.device)
        
        return x

    def load_model(self,choice=0):
        # model = resnet50()
        # model.fc = torch.nn.Linear(2048, 365)
        # dict = torch.load('./resnet50_places365.pth.tar')['state_dict']
        # model.load_state_dict({k.replace('module.', ''): v for k, v in dict.items()})
        model = mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(in_features=1280, out_features=10, bias=True)
        model.load_state_dict(torch.load('./train_checkpoint/MobileNet_epoch_29.pth',map_location=self.device))
        return model
    def predict(self, img_path):
        self.set_seed()
        model = self.load_model().to(self.device)
        model.eval()
        x = self.img_load(img_path=img_path)
        output = model(x)
        probability = torch.nn.functional.softmax(output, dim=1)  # 计算
        prob = probability.max().item()
        max_value, index = torch.max(probability, 1)
        index = index.item()
        return index, prob




if __name__ == '__main__':
    img = Image.open('./test.png')
    scene_predicter = Predicter()
    index, prob = scene_predicter.predict("./test.png")
    print(index)
    # TODO: 修改下读取同片的方式，png格式是32位读取，而jpg是24位深度的
    
    # pred = Predicter()
    # img_path = './test2.jpg'
    # index,prob = pred.predict(img_path=img_path,model_choice=1)
    # model = resnet50()
    # model.fc = torch.nn.Linear(2048, 365)
    # dict = torch.load('./resnet50_places365.pth.tar')['state_dict']
    # model.load_state_dict({k.replace('module.', ''): v for k, v in dict.items()})
    # for key in dict.keys():
    #     key = key.replace('module.','')
    #     print(key)
    # print(dict.items())
    # model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('checkpoint.pt').items()})
    # print(model)
    # 相当于用''代替'module.'。
    # 直接使得需要的键名等于期望的键名。
