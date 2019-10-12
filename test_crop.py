#-*- coding:utf-8 -*- 

import torch
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image

trans = transforms.RandomCrop(109, padding=4)
img = Image.open('/home/data/cy/dataset/fruits-360/Training/Apple Golden 2/0_100.jpg')
print(img.size)
out = trans(img)
print(out.size)
out.save('./out.png')
