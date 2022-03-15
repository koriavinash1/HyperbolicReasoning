import torch
import torch.nn as nn
from torchvision import models
import torch.utils.model_zoo as model_zoo
import os
from utee import misc
from collections import OrderedDict
print = misc.logger.info

model_urls = {
    'stl10': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/stl10-866321e9.pth',
}

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )
        print(self.features)
        print(self.classifier)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def vanilla(n_channel=32, classCount = 10, pretrained=None):
    cfg = [
        # n_channel, 'M',
        # 2*n_channel, 'M',
        # 4*n_channel, 'M',
        # 4*n_channel, 'M',
        n_channel, 'M', 
        n_channel, 'M', 
        2*n_channel, 'M',
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=32*n_channel, num_classes=classCount)
    return model



class DenseNet201(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(DenseNet201, self).__init__()
        
        self.first_conv  = nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet201(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

class DenseNet169(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(DenseNet169, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet169(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

class DenseNet161(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(DenseNet161, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet161(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x


class DenseNet121(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(DenseNet121, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        densenet = models.densenet121(pretrained= isTrained)
        self.features    = densenet.features
        kernelCount = densenet.classifier.in_features
        self.classifier = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x= self.first_conv(x)
        x= self.features(x)
        x= nn.functional.adaptive_avg_pool2d(x,(1,1)).view(x.size(0),-1)
        x= self.classifier(x)
        return x

#===============================================================================================================

class ResNet152(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(ResNet152, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet152(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classCount)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet50(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(ResNet50, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet50(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classCount)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet34(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(ResNet34, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet34(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classCount)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x

class ResNet18(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(ResNet18, self).__init__()
        
        self.resnet = models.resnet18(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classCount)

    def forward(self, x):
        x= self.resnet(x)
        return x


class ResNet101(nn.Module):

    def __init__(self, num_channel = 3, classCount=2, isTrained = True):
        
        super(ResNet101, self).__init__()
        
        self.first_conv  =nn.Sequential(nn.Conv2d(num_channel, 3, kernel_size=3, padding=1),nn.BatchNorm2d(3))
        self.resnet = models.resnet101(pretrained= isTrained)
        self.resnet.avgpool=nn.AdaptiveAvgPool2d((1,1))
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, classCount)

    def forward(self, x):
        x= self.first_conv(x)
        x= self.resnet(x)
        return x
