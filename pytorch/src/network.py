import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import pdb

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0/len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]

# convnet without the last layer
class AlexNetFc(nn.Module):
  def __init__(self, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(AlexNetFc, self).__init__()
    model_alexnet = models.alexnet(pretrained=True)
    self.features = model_alexnet.features
    self.classifier = nn.Sequential()
    for i in range(6):
      self.classifier.add_module("classifier"+str(i), model_alexnet.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_vgg.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features


resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=True)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features

  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

  def get_parameters(self):
    if self.new_cls:
        if self.use_bottleneck:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":self.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":self.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":self.feature_layers.parameters(), "lr_mult":1}, \
                            {"params":self.fc.parameters(), "lr":10}]
    else:
        parameter_list = [{"params":self.parameters(), "lr":1}]
    return parameter_list

vgg_dict = {"VGG11":models.vgg11, "VGG13":models.vgg13, "VGG16":models.vgg16, "VGG19":models.vgg19, "VGG11BN":models.vgg11_bn, "VGG13BN":models.vgg13_bn, "VGG16BN":models.vgg16_bn, "VGG19BN":models.vgg19_bn} 
class VGGFc(nn.Module):
  def __init__(self, vgg_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
    super(VGGFc, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.feature_layers = nn.Sequential(self.features, self.classifier)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.bottleneck.apply(init_weights)
            self.fc.apply(init_weights)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.apply(init_weights)
            self.__in_features = 4096
    else:
        self.fc = model_vgg.classifier[6]
        self.__in_features = 4096

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features

# For SVHN dataset
class DTN(nn.Module):
    def __init__(self):
        super(DTN, self).__init__()
        self.conv_params = nn.Sequential (
                nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(64),
                nn.Dropout2d(0.1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(128),
                nn.Dropout2d(0.3),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                nn.BatchNorm2d(256),
                nn.Dropout2d(0.5),
                nn.ReLU()
                )
    
        self.fc_params = nn.Sequential (
                nn.Linear(256*4*4, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout()
                )

        self.classifier = nn.Linear(512, 10)
        self.__in_features = 512

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                nn.Conv2d(1, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )
        
        self.fc_params = nn.Sequential(nn.Linear(50*4*4, 500), nn.ReLU(), nn.Dropout(p=0.5))
        self.classifier = nn.Linear(500, 10)
        self.__in_features = 500


    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = self.fc_params(x)
        y = self.classifier(x)
        return x, y

    def output_num(self):
        return self.__in_features

class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature, hidden_size):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, hidden_size)
    self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
    self.ad_layer3 = nn.Linear(hidden_size, 1)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()
    self.apply(init_weights)
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = 1.0
    self.max_iter = 10000.0

  def forward(self, x):
    if self.training:
        self.iter_num += 1
    coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
    x = x * 1.0
    x.register_hook(grl_hook(coeff))
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    y = self.ad_layer3(x)
    y = self.sigmoid(y)
    return y

  def output_num(self):
    return 1
  def get_parameters(self):
    return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]
