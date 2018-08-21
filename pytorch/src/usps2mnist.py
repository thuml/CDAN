import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from data_list import ImageList
import os
from torch.autograd import Variable
import loss as loss_func
import numpy as np

class AdversarialLayer(torch.autograd.Function):
  def __init__(self, high_value=1.0):
    self.iter_num = 0
    self.alpha = 10
    self.low = 0.0
    self.high = high_value
    self.max_iter = 10000.0
    
  def forward(self, input):
    self.iter_num += 1
    output = input * 1.0
    return output

  def backward(self, gradOutput):
    self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
    return -self.coeff * gradOutput


class AdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 500)
    self.ad_layer2 = nn.Linear(500,500)
    self.ad_layer3 = nn.Linear(500, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer1.bias.data.fill_(0.0)
    self.ad_layer2.bias.data.fill_(0.0)
    self.ad_layer3.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    x = self.dropout2(x)
    x = self.ad_layer3(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_params = nn.Sequential(
                #nn.Conv2d(1, 20, kernel_size=5),
                nn.Conv2d(3, 20, kernel_size=5),
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

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        y = self.fc2(x)
        return x, y

    def output_num(self):
        return self.__in_features

def train(args, model, ad_net, grl_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch):
    model.train()
    len_source = len(train_loader)
    len_target = len(train_loader1)
    if len_source > len_target:
        num_iter = len_source - 1
    else:
        num_iter = len_target - 1
    
    for batch_idx in range(num_iter):
        if batch_idx % (len_source-1) == 0:
            iter_source = iter(train_loader)    
        if batch_idx % (len_target-1) == 0:
            iter_target = iter(train_loader1)
        data_source, label_source = iter_source.next()
        data_source, label_source = Variable(data_source).cuda(), Variable(label_source).cuda()
        data_target, label_target = iter_target.next()
        data_target = Variable(data_target).cuda()
        optimizer.zero_grad()
        optimizer_ad.zero_grad()
        feature, output = model(torch.cat((data_source, data_target), 0))
        loss = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source)
        #if epoch > 1:
        #    loss += loss_func.CDAN([feature, nn.Softmax(dim=1)(output).detach()], [ad_net], [grl_layer], use_focal=False)
        loss.backward()
        optimizer.step()
        if epoch > 1:
            optimizer_ad.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data_source), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.cpu()[0]))

def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
            data, target = Variable(data).cuda(), Variable(target).cuda()
            feature, output = model(data)
            test_loss += nn.CrossEntropyLoss()(output, target).data.cpu()[0] # sum up batch loss
            pred = output.data.cpu().max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.cpu().view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_ad', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--gpu_id', type=str,
                        help='cuda device id')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id


    source_list = '../data/usps2mnist/usps_train.txt'
    target_list = '../data/usps2mnist/mnist_train.txt'
    test_list = '../data/usps2mnist/mnist_test.txt'

    kwargs = {'num_workers': 1}
    train_loader = torch.utils.data.DataLoader(
        ImageList(open(source_list).readlines(), transform=transforms.Compose([
                           transforms.Resize((28,28)), 
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    train_loader1 = torch.utils.data.DataLoader(
        ImageList(open(target_list).readlines(), transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        ImageList(open(test_list).readlines(), transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    model = LeNet()
    model = model.cuda()
    ad_net = AdversarialNetwork(model.output_num()*10)
    ad_net = ad_net.cuda()
    grl_layer = AdversarialLayer()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0005, momentum=0.9)
    optimizer_ad = optim.SGD(ad_net.parameters(), lr=args.lr_ad, weight_decay=0.0005, momentum=0.9)

    for epoch in range(1, args.epochs + 1):
        if epoch % 6 == 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * 0.5
        train(args, model, ad_net, grl_layer, train_loader, train_loader1, optimizer, optimizer_ad, epoch)
        test(args, model, test_loader)


if __name__ == '__main__':
    main()
