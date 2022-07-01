# -*- coding: utf-8 -*-
# @Time    : 2022-05-02 20:45
# @Author  : Zhikang Niu
# @FileName: utils.py
# @Software: PyCharm
from typing import List, Dict
import os
import shutil

"""
实现文件的位置转移

传入参数：.txt的文档，需要的类别(传入列表)
执行的操作：将所有文件执行操作移动到1~len(类别)-1的文件夹
"""


def file_move(file_path: str, target_list: list, image_path: str, dst_path: str) -> (List, List):
    label_list = []
    image_list = []

    # 读取文件进行处理
    with open(file_path, 'r') as f:
        line = f.readline()
        line_split = line.split()

        label = line_split[-1]
        image = line_split[0]

        if int(label) in target_list:
            img_path = os.path.join(image_path, image)
            shutil.copy(img_path, os.path.join(dst_path,str(label)))

        while line:
            line = f.readline()
            if line:
                line_split = line.split()
                label = line_split[1]
                image = line_split[0]
                if int(label) in target_list:
                    img_path = os.path.join(image_path, image)
                    shutil.copy(img_path, os.path.join(dst_path, str(label)))

    make_dirs(target_list=target_list, move_path="./19_classes/val")
    # make_dirs(target_list=target_list,move_path="./19_classes/train")
    print("finish")


def make_dirs(target_list, move_path: str):
    # 创建文件夹
    for i in target_list:
        if isinstance(i, int):
            if not os.path.exists(os.path.join(move_path, str(i))):
                os.makedirs(os.path.join(move_path, str(i)))
        else:
            if not os.path.exists(os.path.join(move_path, i)):
                os.makedirs(os.path.join(move_path, i))


def class2index(classes_name: list, categories_file="./place365/filelist/categories_places365.txt") -> (List, Dict):
    # 将类名转换为索引
    class2index_dict = {}
    index_list = []
    with open(categories_file, 'r') as f:
        line = f.readline()
        line_split = line.split()
        index = line_split[-1]
        name = line_split[0]
        # 将类名和索引成为字典
        class2index_dict[name] = index
        while line:
            line = f.readline()
            if line:
                line_split = line.split()
                index = line_split[-1]
                name = line_split[0]
            class2index_dict[name] = index
    # 将对应的转换为序列
    for i in classes_name:
        i = "/" + i[0] + "/" + i
        index = class2index_dict.get(i)
        if index is not None:
            index_list.append(int(index))
    return index_list, class2index_dict


if __name__ == '__main__':
    CLASSES_NAME = [
        "airfield", "bookstore", "amusement_park", "coffee_shop", "kitchen",
        "park", "restaurant_patio", "childs_room", "forest/broadleaf", "florist_shop/indoor",
        "campus", "library/indoor", "office", "veterinarians_office", "lawn",
        "bar", "supermarket", "bedroom", "plaza"
    ]
    make_dirs(CLASSES_NAME, './19_classes/val/')
    make_dirs(CLASSES_NAME, './19_classes/train/')
    index_list, class2index_dict = class2index(CLASSES_NAME)
    file_move('./place365/filelist/places365_val.txt',
            target_list=index_list,
            image_path="./place365/val/val_256",
            dst_path="./19_classes/val/")
