#-*- coding:utf-8 -*- 
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from math import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import numpy as np
import cv2
import random


class HeadDataset(Dataset):
    def __init__(self,root_dir,transforms = None):
        self.transforms = transforms
        self.root_dir = root_dir
        self.imgdict = self.get_imgs(self.root_dir)
       
    def __len__(self):
        return len(self.imgdict)

    def __getitem__(self,index):      #----运行顺序
        image_name = self.imgdict.keys()[index]
        image_path = os.path.join(self.root_dir,image_name)
        image_path = image_name
        image = cv2.imread(image_path)
        label = self.imgdict[image_name]
        if self.transforms:
            image = self.transforms(image)
        sample = {"image":image,"label":label}
        return sample

    def get_imgs(self,root_dir):
        imgdict = {}
        num = 0
        dirs = os.listdir(root_dir)
        dirs.sort(key=lambda x: str(x[0]))
        for img_path in dirs:
            if num > 19:
                break
            print(img_path,num)
            cur_path = os.path.join(root_dir, img_path)
            imglist = os.listdir(cur_path)
            for paths in imglist:
                filename = os.path.join(cur_path, paths)
                imgdict[filename] = num
            num += 1
        return imgdict

class Resize(object):
    def __call__(self,image):
       image = cv2.resize(image,(224,224))
       return image

class RandomRotate(object):
    def __init__(self):
        self.degree = random.randint(-90,90)
    def __call__(self,image):
        height,width = image.shape[:2]
        heightNew = int(width * fabs(sin(radians(self.degree))) + height * fabs(cos(radians(self.degree))))
        widthNew = int(height * fabs(sin(radians(self.degree))) + width * fabs(cos(radians(self.degree))))
        matRotation=cv2.getRotationMatrix2D((width/2,height/2),self.degree,1)
        matRotation[0,2] +=(widthNew-width)/2
        matRotation[1,2] +=(heightNew-height)/2
        imgRotation=cv2.warpAffine(image,matRotation,(widthNew,heightNew),borderValue=(0,0,0))
        image = imgRotation
        return image

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # bgr2rgb
        image = image[:,:,(2,1,0)]   #---将通道RGB换成BGR
        image = image.transpose((2, 0, 1))  #---[H,W,C]换成[C,H,W]
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = image/255.
        return image

def show_sample(sample_batch,i):
    image_batch,label = sample_batch["image"],sample_batch["label"]
    grid = make_grid(image_batch)
    plt.imshow(grid.numpy().transpose(1,2,0))
    plt.title(list(label))
    path = "./"+str(i)+".jpg"
    plt.savefig(path)

def loadtraindata(train_dir = None,val_dir = None):
    head_dataset = {}
    train_set = HeadDataset(train_dir, transforms=transforms.Compose([RandomRotate(), Resize(), ToTensor()]))
    #-----torchvision.transforms.Compose([torchvision.transforms.Normalize(),torchvision.transforms.Resize(),torchvision.transforms.ToTensor]) #---前面加上torchvision.transforms则是用torch自带的预处理函数
    #-----只需要一个预处理是直接torchvision.transforms.ToTensor(),多个时需要.Compose()函数

    val_set = HeadDataset(val_dir,transforms=transforms.Compose([RandomRotate(),Resize(),ToTensor()]))
    head_dataset["train"] = train_set
    head_dataset["val"] = val_set
    datasize={"train":len(train_set),"val":len(val_set)}
    dataloaders = {x : DataLoader(head_dataset[x],batch_size = 8,shuffle=True,num_workers = 4) for x in ["train","val"]}
    return dataloaders,datasize

def loadtestdata(test_dir = None):
    test_set = HeadDataset(test_dir,transforms=transforms.Compose([Resize(),ToTensor()]))
    dataloader = DataLoader(test_set,batch_size = 4,shuffle=True,num_workers=4)
    return dataloader
    

''' train_dir = "/home/data/cy/dataset/fruits-360/Training"
val_dir = "/home/data/cy/dataset/fruits-360/Test"
print("begin")
dataloader, datasize = loadtraindata(train_dir=train_dir, val_dir=val_dir)
print(datasize)
print("ok")
for i, sample in enumerate(dataloader['train']):
    print(sample['image'].size())
    show_sample(sample, i)
    break   '''