import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def DAN(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)

def RTN(input_list, softmax_layer, silence_layer, entropy_trade_off=0.1, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    output = input_list[1]
    feature = input_list[0]
    batch_size = output.size(0) // 2

    softmax_output = softmax_layer(output)
    entropy_loss = EntropyLoss(softmax_output.narrow(0,0,batch_size))

    #op_out = torch.bmm(output.unsqueeze(2), feature.unsqueeze(1))
    #op_out = op_out.view(-1, output.size(1) * feature.size(1))
    #feature_dim = feature.size(1)
    #class_num = output.size(1)
    #pool_out = nn.MaxPool2d(2, stride=2)(op_out.view(-1, 1, class_num, feature_dim))
    #pool_out = pool_out.view(batch_size*2, -1)
    source = feature.narrow(0, 0, batch_size)
    target = feature.narrow(0, batch_size, batch_size)
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size) + entropy_trade_off * entropy_loss
   
    

def JAN(source_list, target_list, kernel_muls=[2.0, 2.0], kernel_nums=[5, 1], fix_sigma_list=[None, 1.68]):
    batch_size = int(source_list[0].size()[0])
    layer_num = len(source_list)
    joint_kernels = None
    for i in range(layer_num):
        source = source_list[i]
        target = target_list[i]
        kernel_mul = kernel_muls[i]
        kernel_num = kernel_nums[i]
        fix_sigma = fix_sigma_list[i]
        kernels = guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        if joint_kernels is not None:
            joint_kernels = joint_kernels * kernels
        else:
            joint_kernels = kernels
    
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1)%batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += joint_kernels[s1, s2] + joint_kernels[t1, t2]
        loss -= joint_kernels[s1, t2] + joint_kernels[s2, t1]
    return loss / float(batch_size)


def MADA(input_list, ad_net_list, grl_layer, use_gpu=True):
    softmax_output = input_list[1]
    feature = input_list[0]
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    class_num = softmax_output.size(1)
    batch_size = softmax_output.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    
    dc_loss = 0
    for i in range(class_num):
        ad_out = ad_net_list[i](grl_layer(op_out.narrow(1, i, 1).contiguous().view(-1, feature.size(1))))
        dc_loss += nn.BCELoss()(ad_out, dc_target)
    return dc_loss


def CADA(input_list, ad_net, grl_layer, use_focal=True, use_gpu=True):
    softmax_output = input_list[1]
    feature = input_list[0]
    op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
    ad_out = ad_net(grl_layer(op_out.view(-1, softmax_output.size(1) * feature.size(1))))
    batch_size = softmax_output.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    if use_focal:
        source_focal = ad_out.data.narrow(0,0,batch_size).clone()
        target_focal = 1.0 - ad_out.data.narrow(0,batch_size,batch_size).clone()
        focal_weight = torch.exp(torch.cat((source_focal, target_focal), dim=0))
        focal_weight = focal_weight / (math.e-1)
    return nn.BCELoss(weight=focal_weight)(ad_out, dc_target)

    
def CADA_R(input_list, ad_net, grl_layer, rman_layer, use_focal=True, use_gpu=True):
    softmax_output = input_list[1]
    feature = input_list[0]
    rman_out_list = rman_layer.forward([feature, softmax_output])
    rman_out = rman_out_list[0]
    for rman_single_out in rman_out_list[1:]:
        rman_out = torch.mul(rman_out, rman_single_out)
    ad_out = ad_net(grl_layer(rman_out.view(-1, rman_out.size(1))))
    batch_size = softmax_output.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    if use_focal:
        source_focal = ad_out.data.narrow(0,0,batch_size).clone()
        target_focal = 1.0 - ad_out.data.narrow(0,batch_size,batch_size).clone()
        focal_weight = torch.exp(torch.cat((source_focal, target_focal), dim=0))
        focal_weight = focal_weight / (math.e-1)
    return nn.BCELoss(weight=focal_weight)(ad_out, dc_target)


def DANN(features, ad_net, grl_layer, use_gpu=True):
    ad_out = ad_net(grl_layer(features))
    batch_size = ad_out.size(0) // 2
    dc_target = Variable(torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
    return nn.BCELoss()(ad_out, dc_target)
   
