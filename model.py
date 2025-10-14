import torch
import torch.nn as nn
import torch.nn.init
from torch.nn import Module, Conv2d, Parameter, Softmax
use_cuda = torch.cuda.is_available()
from config import args


#Model
class MyNet(nn.Module):
    def __init__(self,input_dim, channel1=100, channel2=100, channel3=100, channel4=100, channel5=100, channel6=100):
        super(MyNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(input_dim, channel1, 3, stride=1,  padding=1),
                                   nn.BatchNorm2d(channel1),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(channel1, channel2, 3, stride=1,  padding=1),
                                   nn.BatchNorm2d(channel2),
                                   nn.ReLU())

        self.conv2a = nn.Sequential(nn.Conv2d(channel2, channel3, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(channel3),
                                   nn.ReLU())

        self.conv2b = nn.Sequential(nn.Conv2d(channel3, channel4, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(channel4),
                                   nn.ReLU())

        self.conv2c = nn.Sequential(nn.Conv2d(channel4, channel5, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(channel5),
                                   nn.ReLU())

        self.conv2d = nn.Sequential(nn.Conv2d(channel5, channel6, 3, stride=1, padding=1),
                                   nn.BatchNorm2d(channel6),
                                   nn.ReLU())

        self.conv3 = nn.Sequential(nn.Conv2d(channel6, args.outClust, 1, stride=1),
                                   nn.BatchNorm2d(args.outClust))

        self.drop = nn.Sequential(nn.Dropout2d(0.1, False))



    def forward(self, x):
        out1 = self.conv1(x)

        out2 = self.conv2(out1)

        out3 = self.conv2a(out2)

        out4 = self.conv2b(out3)

        out5 = self.conv2c(out4)

        out6 = self.conv2d(out5)

        drop4 = self.drop(out6)

        out = self.conv3(drop4)
        #drop = self.drop(out)

        return out #torch.softmax(out, dim=1)

