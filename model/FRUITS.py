#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn

class Fruits(nn.Module):
    def __init__(self,num_classes):
        super(Fruits, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(  #---(3,100,100)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5,
                    stride=1, padding=2),  #(64,100,100)
            nn.ReLU(),  #-----mnist数据集图片本身比较小,因此构造的out_channels较小,
                        #----而水果的图片稍微会大一点，channels需要大一点
            nn.MaxPool2d(kernel_size=2)  #(batch_size,64,50,50)   
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, 1, 2),   #(batch_size,128,50,50)
            nn.ReLU(),
            nn.MaxPool2d(2)     #(batch_size,128,25,25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128* 25* 25, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.out = nn.Sequential(
            nn.Linear(512, num_classes),
        )             

        # self.softmax = nn.Softmax(dim=-1)
#----网络的输出mnist是16,32,在这里是64,128

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)   #----x.size(0)为batch_size  (batch_size,32*7*7),本来是(batch_size,32,7,7)
        x = self.fc1(x)
        x = self.fc2(x)
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

