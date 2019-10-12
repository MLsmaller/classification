#-*- coding:utf-8 -*-
import os
import torch
from MNIST import Mnist
from PIL import Image
import cv2
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import random

DOWNLOAD_MNIST = False
model_path = "/home/gcl/squeezeface/mnist/weights/mnist_10000.pth"
test_img = './test_imgs/3.jpg'


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    net = Mnist()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    net=net.to(device)
    # img = Image.open(test_img)
    img = cv2.imread(test_img)
    assert (img is not None), "error read img"
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    tva = [[(255-img[x, y]) * 1.0/255.0  for y in range(w)] for x in range(h)]  #---遍历每个像素进行操作
    # tva = [[(img[x, y]) * 1.0/255.0  for y in range(w)] for x in range(h)]  #---针对mnist数据
    img = np.array(tva)
    img = img[:,:, np.newaxis]
    # img = img[:,:, (2, 1, 0)]      #----灰度图,无RGB/BGR
    img = img.transpose((2, 0, 1))   
    img = torch.from_numpy(img).type(torch.FloatTensor)   #---To torch.tensor
    img = img.unsqueeze(0)  #---=(1,1,28,28)
    
    img = img.cuda()
    output = net(img)
    print(output.size())
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)    #---Tensor -> Variable
    prob=prob.cpu().numpy()  #----Variable -> numpy 因是在cpu上显示,因为.cpu() 
    pred = np.argmax(prob, 1)

    print("预测的数字为: {} ".format(pred.item()))
    
def read_mnist():
    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
)
    index = random.randint(1, 10000)  
    test_img = test_data.test_data[index]
    test_label = test_data.test_labels[index]
    print("真实图片数字为: {}".format(test_label.numpy()))
    img = test_img.numpy()   #----Tensor -> numpy
    cv2.imwrite('./test_imgs/mnist.png', img)
    

if __name__ == "__main__":
    
    read_mnist()
    main()
    # a = torch.randn((2, 3))
    # a=a.unsqueeze(0)   #----tensor.unsqueeze()对于tensor对象使用
    # b = a[:,np.newaxis,:]