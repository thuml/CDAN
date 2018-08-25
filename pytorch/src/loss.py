import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

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
   
