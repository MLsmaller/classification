#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

class Mnist(nn.Module):
    #---num_classes为分类数量
    def __init__(self,num_classes):
        super(Mnist, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(  #---(batch_size,1,28,28,)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5,
                    stride=1, padding=2),  #(16,28,28)
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)  #(batch_size,16,14,14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),   #(batch_size,32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)     #(batch_size,32,7,7)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
        )

        self.out = nn.Linear(64 * 7 * 7, num_classes)   #---搭建网络,输入输出
        # self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)   #----x.size(0)为batch_size  (batch_size,32*7*7),本来是(batch_size,32,7,7)
        output = self.out(x)
        # output = self.softmax(output)
        return output

if __name__ == "__main__":
    print("haha")
    # a = torch.randn(50, 1, 28, 28)
    # net = Mnist()
    # output = net(a)
    # value,index = torch.max(output, 1)  #--返回每一行中最大值元素和索引,0则是每一列
    # print(output.size())
    # print(index)
    # print(index.data)  #---tensor.data()取出tensor中的数据
    # print(index.data.squeeze())   #---squeeze()函数去掉维度为1的部分

