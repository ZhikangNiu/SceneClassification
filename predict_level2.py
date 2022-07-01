# -*- coding: utf-8 -*-
# @Time    : 2022-06-25 18:59
# @Author  : Zhikang Niu
# @FileName: acc_level2.py
# @Software: PyCharm

from PIL import Image
import random
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torchvision.models import  mobilenet_v3_large
from PIL import Image



class Predicter():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = mobilenet_v3_large(pretrained=True)
        self.model.classifier[3] = nn.Linear(in_features=1280, out_features=25, bias=True)
        self.model.load_state_dict(torch.load('./MobileNetv3_large_epoch_28.pth', map_location=self.device)["model_state_dict"])
        self.model.to(self.device)


    def set_seed(self, seed=66):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def img_load(self, img_path):
        # TODO: 看下opencv奇奇怪怪的问题
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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

    def predict(self, img_path,threshold = 0.6):
        self.set_seed()
        model = self.model
        model.eval()
        x = self.img_load(img_path=img_path)
        output = model(x)
        probability = torch.nn.functional.softmax(output, dim=1)  # 计算
        prob = probability.max().item()
        max_value, index = torch.max(probability, 1)
        index = index.item()
        # prob 是概率，index 是索引
        if prob < threshold:
            index = 25
            # 25为未知类
        return index, prob


if __name__ == '__main__':
    img = Image.open('./test.png')
    scene_predicter = Predicter()
    index, prob = scene_predicter.predict("./test.png")
    print(index)
    print(prob)
    # TODO: 修改下读取同片的方式，png格式是32位读取，而jpg是24位深度的

