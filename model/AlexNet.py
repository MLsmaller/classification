#-*- coding:utf-8 -*- 

import torch
import torch.nn as nn
from torch.autograd import Variable

num_classes = 20

class AlexNet(nn.Module):

    def __init__(self, ngpus):
        super(AlexNet, self).__init__()
        self.ngpu = ngpus
        self.features = nn.Sequential(    #----3x100x100
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            # raw kernel_size=11, stride=4, padding=2. For use img size 224 * 224.
            nn.ReLU(inplace=True),         #-----64x50x50
            nn.MaxPool2d(kernel_size=3, stride=2),  #----64x24x24
            nn.Conv2d(64, 192, kernel_size=5, padding=2),   #192x24x24
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          #192x11x11
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  #384x11x11
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  #256x11x11
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  #256x11x11
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),     #256x5x5
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 1*1, 4096),   #256x(size/20)x(size/20)  size为输入图像大小
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
  
    def forward(self, inputs):
        inputs = self.features(inputs)
        # print(inputs.size())
        inputs = inputs.view(-1, 256 *1*1)
        inputs = self.classifier(inputs)
        return inputs
        
if __name__ == "__main__":
    net = AlexNet(3)
    input = Variable(torch.randn(50, 3, 50, 50))
    y = net(input)
    
    print(y.size())