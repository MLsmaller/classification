#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import transforms

from model.MNIST import Mnist
from model.CIfar import cifar
from model.ResNet import ResNet18

from model.letnet import Letnet
from utils.factory import dataset_factory

import os
import time
import cv2

import numpy as np
import argparse,random

parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpu', type=bool, choices=[True, False], default=True,
                    help="是否使用多gpu")
parser.add_argument('--train_dir', type=str, default="/home/data/cy/dataset/fruits-360/Training",
                    help="训练集数据的路径")
parser.add_argument('--val_dir', type=str, default="/home/data/cy/dataset/fruits-360/Test",
                    help="验证集数据的路径")
parser.add_argument('--num_classes', type=int, default=120,
                    help="分类种类")
parser.add_argument('--BATCH_SIZE', type=int, default=128,
                    help="训练集batch_size")
parser.add_argument('--EPOCH', type=int, default=20)
parser.add_argument('--use_cuda', type=bool, choices=[True, False], default=True,
                    help='是否使用gpu')
parser.add_argument('--dataset', type=str, default='fruits',
                    help='数据集类型')
parser.add_argument('--num_workers', type=int, default=4,
                    help='进程数量')
parser.add_argument('--LR', type=float, default=0.001,
                    help='初始学习率')
parser.add_argument('--weights', type=str, default="/home/data/cy/projects/classification/fruits/weights",
                    help="模型保存位置" )
parser.add_argument('--gpu_ids', type=str, default="2,3",
                    help="gpu id ")
args = parser.parse_args()


gpu_ids = [int(x) for x in list(args.gpu_ids) if x.isalnum()]

device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')  #----这里device需要指定为gup_ids[0],因为默认为gup:0,如果gpu_ids中不包含0,,则会报错

train_dataset, val_dataset = dataset_factory(args.dataset, args.train_dir, args.val_dir)

train_loader = Data.DataLoader(train_dataset, args.BATCH_SIZE,
                                num_workers=args.num_workers,
                                shuffle=True)

VAL_BATCH_SIZE = args.BATCH_SIZE    
print("train batch and val batch is {}".format(args.BATCH_SIZE, VAL_BATCH_SIZE))
val_loader = Data.DataLoader(val_dataset, VAL_BATCH_SIZE,
                                num_workers=args.num_workers,
                                shuffle=True)

train_nums = len(train_dataset)
test_nums = len(val_dataset)
datasize = {'train': train_nums, 'val': test_nums}
print(datasize)
data_loader = {'train': train_loader, 'val': val_loader}

net = ResNet18(args.num_classes)

print(net)
print("batch_size is {}".format(args.BATCH_SIZE))
print("val batch_size is {}".format(VAL_BATCH_SIZE))
if args.multi_gpu:   #----multi-gpu  ,当数据集非常小时,mult-gpu可能花费实际大于single-gpu
    net = nn.DataParallel(net, device_ids=gpu_ids).to(device)
else:
    net = net.to(device)
# net = net.to(device)
    
optimizer = torch.optim.Adam(net.parameters(), lr=args.LR)   
print(optimizer)
criterion = nn.CrossEntropyLoss()  
step_size = 7
scheduler = lr_scheduler.StepLR(optimizer ,step_size = step_size, gamma = 0.1)  

print("step_size is {}".format(step_size))

save_path = '/home/data/cy/projects/classification/fruits/weights/'

def train():
    iteration = 0
    for epoch in range(args.EPOCH):  #----一个epoch训练一个完整的数据集
        begin = time.time()
        print('Epoch {}/{}'.format(epoch, args.EPOCH - 1))
        print('-'*20)
        #----在一个epoch内,对数据进行批量读入(每次读入是一个iteration),然后训练
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
                scheduler.step()
            else:
                net.eval()
            running_loss = 0.0
            running_corrects = 0

            for step, samples in enumerate(data_loader[phase]):
                inputs = samples["image"].to(device)  
                labels = samples["label"].to(device)  
                optimizer.zero_grad()    #---梯度(网络参数)清零,每一个batch都不一样
                # print(inputs.size())
                with torch.set_grad_enabled(phase == 'train'):  
                    outputs = net(inputs)                     
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)   
                    if phase == 'train':
                        loss.backward()     
                        optimizer.step()    
                
                running_loss += loss.item() * inputs.size(0)    #--累计当前batch的loss
                running_corrects += torch.sum(preds == labels.data) #--累计当前Batch计算正确的个数
                
                epoch_loss = running_loss / datasize[phase]
                epoch_acc = running_corrects.double() / datasize[phase]
                
                if iteration != 0 and iteration % 5000 == 0:
                    print('saving state,iter:', iteration)
                    file = 'fruits_{0}.pth'.format(iteration)
                    torch.save(net.state_dict(),os.path.join(save_path,file))
                iteration += 1
            print('  {} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
        end = time.time()
        print("  Epoch {} cost {:.4f} s".format(epoch, end-begin))
        
         
if __name__ == "__main__":
    train()
    print("done")
    
