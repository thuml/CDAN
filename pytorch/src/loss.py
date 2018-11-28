import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def CADA(input_list, ad_net, entropy=None, coeff=None):
    softmax_output = input_list[1]
    feature = input_list[0]
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(-coeff))
        entropy = torch.exp(-entropy)
        return torch.mean(entropy * nn.BCELoss(size_average=False)(ad_out, dc_target))
    else:
        return nn.BCELoss()(ad_out, dc_target) 

    
def CADA_R(input_list, ad_net, random_layer, entropy=None, coeff=None):
    softmax_output = input_list[1]
    feature = input_list[0]
    random_out_list = random_layer.forward([feature, softmax_output])
    random_out = random_out_list[0]
    for random_single_out in random_out_list[1:]:
        random_out = torch.mul(random_out, random_single_out)
    ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(-coeff))
        entropy = torch.exp(-entropy)
        return torch.mean(entropy * nn.BCELoss(size_average=False)(ad_out, dc_target))
    else:
        return nn.BCELoss()(ad_out, dc_target) 


def DANN(features, ad_net, weight):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)
