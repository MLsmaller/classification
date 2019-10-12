#-*- coding:utf-8 -*-
import torch
from .fruits import fruitsData
import torchvision
from math import * 
import cv2 
import random
import numpy as np
import torchvision.transforms as transforms

#-----输入图像预处理---

class RandomRotate(object):  #---有object表示继承python中的Object类,这样子类会有较多的对象可以操作,不写则不继承
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

class Resize(object):
    def __init__(self, size):
        self.size = size
    def __call__(self,image):
       image = cv2.resize(image, self.size)
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

def dataset_factory(dataset,train_dir,val_dir):
    if dataset == 'fruits':
        size = (224, 224)
        train_dataset = fruitsData(train_dir, transforms=transforms.Compose(
                                    [transforms.RandomCrop(100, padding=4),  #---使用自带的transforms.()等预处理函数时
                                     transforms.RandomHorizontalFlip(),  #输入图像需要是PIL Image的格式,如果是ndarray格式,需要转换img_PIL=Image.fromarray(img)
                                     transforms.Resize(size),
                                     transforms.ToTensor()]))
        val_dataset = fruitsData(val_dir, transforms=transforms.Compose(
                                    [transforms.Resize(size),
                                     transforms.ToTensor()]))
    
    return train_dataset, val_dataset
    

if __name__ == "__main__":
    test_path = "/home/data/cy/dataset/fruits-360/Training/Apple Braeburn/0_100.jpg"
    img = cv2.imread(test_path)
    net = Resize((100, 100))
    img1 = net(img)
    print(img1.shape)
    
    