#-*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision 
from torch.autograd import Variable
from torchvision import transforms

from model.MNIST import Mnist
from model.CIfar import cifar
from model.ResNet import ResNet18
import os
import time

gpu_ids = [int(x) for x in range(2, 4)]

EPOCH = 10
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = True
use_cuda = True
val_batch_size = BATCH_SIZE    #--整除,向下取整
num_classes = 10

device = torch.device('cuda:{}'.format(gpu_ids[0]) if torch.cuda.is_available() else 'cpu')  #----这里device需要指定为gup_ids[0],因为默认为gup:0,如果gpu_ids中不包含0,,则会报错

training_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((200,200)),
        torchvision.transforms.ToTensor()]),
    download=DOWNLOAD_MNIST,
)

# print(training_data.train_data.size())
# print(training_data.train_labels.size())

train_nums = len(training_data.targets)
train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)


test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((200,200)),
        torchvision.transforms.ToTensor()]),
    download=DOWNLOAD_MNIST
)
# print(test_data.test_data.size())  #----(60000,28,28) 需要加个通道

test_nums = len(test_data.targets)
datasize = {'train': train_nums, 'val': test_nums}
print(datasize)

val_loader = Data.DataLoader(dataset=test_data, batch_size=val_batch_size, shuffle=True)
data_loader = {'train': train_loader, 'val': val_loader}


# net = Mnist(10)
net = ResNet18(10)

print(net)
print("batch_size is {}".format(BATCH_SIZE))
if torch.cuda.device_count() > 1:   #----multi-gpu  ,当数据集非常小时,mult-gpu可能花费实际大于single-gpu
    net = nn.DataParallel(net, device_ids=gpu_ids).to(device)
else:
    net = net.to(device)
# net = net.to(device)
    
optimizer = torch.optim.Adam(net.parameters(), lr=LR)   #---torch.optim.Adam/SGD不同的梯度下降规则,对网络w和b更新
criterion = nn.CrossEntropyLoss()  #---描述两个概率分布(网络输出概率分布/标签)之间的距离,越小就越接近,经过softmax()函数后得到概率分布(不同类别的),从而进行多分类

save_path = './mnist/weights'

def train():
    iteration = 0
    for epoch in range(EPOCH):  #----一个epoch训练一个完整的数据集
        begin = time.time()
        print('Epoch {}/{}'.format(epoch, EPOCH - 1))
        print('-'*20)
        #----在一个epoch内,对数据进行批量读入(每次读入是一个iteration),然后训练
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            running_loss = 0.0
            running_corrects = 0

            for step, (inputs, labels) in enumerate(data_loader[phase]):
                inputs = inputs.to(device)  #----batch_size为50(50,1,28,28) ,Variable()默认不求导
                labels = labels.to(device)  #---每次for循环取出50个数据进行输入
                optimizer.zero_grad()    #---梯度(网络参数)清零,每一个batch都不一样

                # print(inputs.size())
                with torch.set_grad_enabled(phase == 'train'):  #---torch.set_grad_enabled(False)下的tensor及新节点不求导
                    outputs = net(inputs)                       #---训练集需要求导,验证集不用
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)   #---当前batch的loss
                    if phase == 'train':
                        loss.backward()      #--loss反向传播
                        optimizer.step()     #---根据网络反向传播的梯度信息来更新网络参数,从而降低loss
                
                running_loss += loss.item() * inputs.size(0)    #--累计当前batch的loss
                running_corrects += torch.sum(preds == labels.data) #--累计当前Batch计算正确的个数
                
                epoch_loss = running_loss / datasize[phase]
                epoch_acc = running_corrects.double() / datasize[phase]
                
                if iteration != 0 and iteration % 5000 == 0:
                    print('saving state,iter:', iteration)
                    file = 'mnist_{0}.pth'.format(iteration)
                    torch.save(net.state_dict(),os.path.join(save_path,file))
                iteration += 1
            print('  {} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))
        end = time.time()
        print("  Epoch {} cost {:.4f} s".format(epoch, end-begin))
        
         

if __name__ == "__main__":
    train()
    print("done")
