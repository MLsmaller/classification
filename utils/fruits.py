#-*- coding:utf-8 -*-
import torch.utils.data as data
import torchvision.transforms
import torch
import torch.nn as nn
import os, time
import numpy as np
import cv2
import random
from PIL import Image

#-----自定义dataloader类

class fruitsData(data.Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir         #----用以查找图像路径
        self.transforms = transforms      #----图像预处理
        self.imgdict=self.get_imgs(self.root_dir) #----在__init__这里要得到图像数据()和对应的标签,以便调用__getitem__()函数时可以得到相应的数据
        
    def __len__(self):              #---返回图像数据集的长度
        return len(self.imgdict)     

    def __getitem__(self, index):  #---给出一个索引要返回数据以及标签
        sample = self.pull_item(index)   #---调用pull_item来返回数据,以便后面调用和检查
        return sample

    
    def get_imgs(self,root_dir):  
        imgdict = {}
        num = 0
        dirs = os.listdir(root_dir)
        dirs.sort(key=lambda x: str(x))
        for img_path in dirs:
            # if num > 19:
            #     break
            print("{} the labels is {} ".format(img_path, num))
            cur_path = os.path.join(root_dir, img_path)
            imglist = os.listdir(cur_path)
            for paths in imglist:
                filename = os.path.join(cur_path, paths)
                imgdict[filename] = int(num)
            num += 1
        return imgdict

    def pull_item(self, index):
        img_path = list(self.imgdict.keys())[index]  #---py3中dict.keys()返回的不是list类型
        # img = cv2.imread(img_path)
        img = Image.open(img_path)
        label = self.imgdict[img_path]
        if self.transforms:
            img = self.transforms(img)
        sample = {'image': img, "label": label}
        return sample


if __name__ == "__main__":
    train_dir = "/home/data/cy/dataset/fruits-360/Training"
    dataset = fruitsData(train_dir, None)   #----初始化类一定是根据__init__中的参数来初始化类
    nums = dataset.__len__()  #----调用__len__()函数
    sample = dataset.pull_item(2)
    # sample = dataset.__getitem__(2)
    print(sample)
