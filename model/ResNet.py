import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride = 1,need_shortcut = False):
        super(ResidualBlock,self).__init__()
        self.right = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size = 3,stride = stride,padding = 1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(True),
            nn.Conv2d(outchannels,outchannels,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm2d(outchannels)
         )
        if need_shortcut:
            self.short_cut = nn.Sequential(
                nn.Conv2d(inchannels,outchannels,kernel_size = 1,stride = stride),
                nn.BatchNorm2d(outchannels)
            )
        else:
            self.short_cut = nn.Sequential()
    
    def forward(self,x):
        out = self.right(x)
        out += self.short_cut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18,self).__init__()
        self.pre_layer = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.block_1 = ResidualBlock(64,64,stride=1,need_shortcut=True)
        self.block_2 = ResidualBlock(64,64,stride=1,need_shortcut=False)
        self.block_3 = ResidualBlock(64,128,stride=2,need_shortcut=True)
        self.block_4 = ResidualBlock(128,128,stride=1,need_shortcut=False)
        self.block_5 = ResidualBlock(128,256,stride=2,need_shortcut=True)
        self.block_6 = ResidualBlock(256,256,stride=1,need_shortcut=False)
        self.block_7 = ResidualBlock(256,512,stride=2,need_shortcut=True)
        self.block_8 = ResidualBlock(512,512,stride=1,need_shortcut=False)
        self.avepool = nn.AvgPool2d(kernel_size=7,stride=1)
        self.fc = nn.Linear(512,num_classes)
        self.num_classes = num_classes

    def forward(self,x):
        out = self.pre_layer(x)
        out = self.block_2(self.block_1(out))
        out = self.block_4(self.block_3(out))
        out = self.block_6(self.block_5(out))
        out = self.block_8(self.block_7(out))
        out = self.avepool(out)
        out = out.view(-1,self.num_flatters(out))
        return self.fc(out)

    def num_flatters(self,x):
        sizes = x.size()[1:]
        result = 1
        for size in sizes:
            result *= size
        return result 

if __name__ == "__main__":
    net = ResNet18(120)
    x = torch.randn(1,3,100,100)
    print(net(x).size())