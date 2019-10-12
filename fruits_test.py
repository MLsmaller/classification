#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.ResNet import ResNet18
from model.FRUITS import Fruits
from utils.factory import *

import cv2 
import numpy as np
import torch.nn.functional as F
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="/home/data/cy/dataset/fruits-360/Training",
                     help="训练集数据目录")
parser.add_argument('--test_dir', type=str, default="/home/data/cy/dataset/fruits-360/Test",
                     help="训练集数据目录")
parser.add_argument('--num_classes', type=int, default=120,
                     help="分类数目")
parser.add_argument('--model_path', type=str, default="/home/cy/projects/github/project_piano/classification/mnist/weights/mnist_10000.pth",
                     help="模型存储路径")                                                       
parser.add_argument('--test_path', type=str, default="./test_imgs",
                     help="测试图片目录") 
args=parser.parse_args()

num_class = []
dirs = os.listdir(args.train_dir)
dirs.sort(key=lambda x: str(x))
for kind in dirs:
    num_class.append(kind)
    
test_list = os.listdir(args.test_path)
img_list = [os.path.join(args.test_path, x) for x in test_list]
    

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    
    model = ResNet18(args.num_classes)
    model = nn.DataParallel(model)   #----训练时模型使用multi-gpu,测试时也要使用
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)   #---use cuda

    for path in img_list:
        img = cv2.imread(path)
        assert (img is not None), "error read img"
        print(path)
        img = cv2.resize(img, (224, 224))
        tensor = ToTensor()
        img = tensor(img)
        img = img.unsqueeze(0)  
        
        # print(img.size())
        img = img.to(device)

        output = model(img)

        prob = F.softmax(output, dim=1)   #----按行softmax,行的和概率为1,每个元素代表着概率
        print("prob的size为{}".format(prob.size()))
        prob = Variable(prob)
        prob = prob.cpu().numpy()
        pred = np.argmax(prob, 1)
        index = pred.item()
        print("最有可能的类为: {}".format(num_class[index]))
        print("\n")


if __name__ == "__main__":
    main()
    # a = np.array([[1, 2, 7],
    #               [9, 8, 5]])
    # b = np.array([[4, 3, 8]])
    # print(np.argmax(b))  #返回int
    # print(np.argmax(b).item())  #int对象的item()是本身,numpy.item()返回numpy里面的数值
    # print(type(np.argmax(b, 1)))  #返回numpy
    # print(np.argmax(a))      #---排成一列后寻找
    # print(np.argmax(a, 1))   #---按行比较输出最大值索引
    # print(np.argmax(a, 0))  #---按列比较输出最大值索引
    