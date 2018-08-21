import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList
from torch.autograd import Variable

optim_dict = {"SGD": optim.SGD}

def image_classification_predict(loader, model, gpu=True):
    start_test = True
    iter_val = iter(loader)
    for i in range(len(loader)):
        data = iter_val.next()
        inputs = data[0]
        if gpu:
            inputs = Variable(inputs.cuda())
        else:
            inputs = Variable(inputs)
        outputs = model(inputs)
        if start_test:
            all_output = outputs.data.cpu().float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.cpu().float()), 0)
    _, predict = torch.max(all_output, 1)
    return all_output, predict

def image_classification_test(loader, model, gpu=True):
    
    start_test = True
    iter_test = iter(loader)
    for i in range(len(loader)):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        outputs = model(inputs)
        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.float()), 0)
            all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label) / float(all_label.size()[0])
    return accuracy


def transfer_classification(config):
    prep_train  = prep.image_train_cifar()
    prep_test = prep.image_test_cifar()
               
    ## set loss
    class_criterion = nn.CrossEntropyLoss()
    loss_config = config["loss"]
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}

    ## prepare data
    data_config = config["data"]
    source_list = ImageList(open(data_config["list_path"]["source"]).readlines(), transform=prep_train)
    source_loader = util_data.DataLoader(source_list, batch_size=data_config["batch_size"]["train"], shuffle=True, num_workers=2)
    target_list = ImageList(open(data_config["list_path"]["target"]).readlines(), transform=prep_train)
    target_loader = util_data.DataLoader(target_list, batch_size=data_config["batch_size"]["train"], shuffle=True, num_workers=2)
    test_list = ImageList(open(data_config["list_path"]["test"]).readlines(), transform=prep_train)
    test_loader = util_data.DataLoader(test_list, batch_size=data_config["batch_size"]["test"], shuffle=False, num_workers=2)
    class_num = 10

    ## set base network
    net_config = config["network"]
    base_network = network.network_dict[net_config["name"]]()
    if net_config["use_bottleneck"]:
        bottleneck_layer = nn.Linear(base_network.output_num(), net_config["bottleneck_dim"])
        classifier_layer = nn.Linear(bottleneck_layer.out_features, class_num)
    else:
        classifier_layer = nn.Linear(base_network.output_num(), class_num)

    ## initialization
    if net_config["use_bottleneck"]:
        bottleneck_layer.weight.data.normal_(0, 0.005)
        bottleneck_layer.bias.data.fill_(0.1)
        bottleneck_layer = nn.Sequential(bottleneck_layer, nn.ReLU(), nn.Dropout(0.5))
    classifier_layer.weight.data.normal_(0, 0.01)
    classifier_layer.bias.data.fill_(0.0)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        if net_config["use_bottleneck"]:
            bottleneck_layer = bottleneck_layer.cuda()
        classifier_layer = classifier_layer.cuda()
        base_network = base_network.cuda()


    ## collect parameters
    if net_config["use_bottleneck"]:
        parameter_list = [{"params":base_network.parameters(), "lr":1}, {"params":bottleneck_layer.parameters(), "lr":1}, {"params":classifier_layer.parameters(), "lr":1}]
    else:
        parameter_list = [{"params":base_network.parameters(), "lr":1}, {"params":classifier_layer.parameters(), "lr":1}]

    ## add additional network for some methods

    if loss_config["name"] == "DANN":
        if net_config["use_bottleneck"]:
            feature_dim = bottleneck_layer[0].out_features
        else:
            feature_dim = base_network.output_num()
        ad_layer1 = nn.Linear(feature_dim, 1024)
        ad_layer2 = nn.Linear(1024,1024)
        ad_layer3 = nn.Linear(1024, 1)
        ad_layer1.weight.data.normal_(0, 0.01)
        ad_layer2.weight.data.normal_(0, 0.01)
        ad_layer3.weight.data.normal_(0, 0.3)
        ad_layer1.bias.data.fill_(0.0)
        ad_layer2.bias.data.fill_(0.0)
        ad_layer3.bias.data.fill_(0.0)
        ad_net = nn.Sequential(ad_layer1, nn.ReLU(), nn.Dropout(0.5), ad_layer2, nn.ReLU(), nn.Dropout(0.5), ad_layer3, nn.Sigmoid())
        gradient_reverse_layer = network.AdversarialLayer()
        if use_gpu:
            ad_net = ad_net.cuda()
        parameter_list.append({"params":ad_net.parameters(), "lr":0.1})
    elif loss_config["name"] == "JAN":
        softmax_layer = nn.Softmax(dim=1)
        if use_gpu:
            softmax_layer = softmax_layer.cuda()
        
           
 
    ## set optimizer
    optim_parameters = {"lr":1.0, "momentum":0.9, "weight_decay":0.0005, "nesterov":True}
    optimizer = optim.SGD(parameter_list,  **optim_parameters)
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])

    lr_scheduler = lr_schedule.schedule_dict["multistep"]


    ## train   
    len_train_source = len(source_loader) - 1
    len_train_target = len(target_loader) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    epoch_num = 0
    for i in range(config["num_iterations"]):
        ## test in the train
        if i % config["test_interval"] == 0:
            base_network.train(False)
            classifier_layer.train(False)
            if net_config["use_bottleneck"]:
                bottleneck_layer.train(False)
                temp_acc = image_classification_test(test_loader, nn.Sequential(base_network, bottleneck_layer, classifier_layer), gpu=use_gpu)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_model = nn.Sequential(base_network, bottleneck_layer, classifier_layer)
            else:
                temp_acc = image_classification_test(test_loader, nn.Sequential(base_network, classifier_layer), gpu=use_gpu)
                if temp_acc > best_acc:
                    best_acc = temp_acc
                    best_model = nn.Sequential(base_network, classifier_layer)
            print temp_acc

        loss_test = nn.BCELoss()
        ## train one iter
        base_network.train(True)
        if net_config["use_bottleneck"]:
            bottleneck_layer.train(True)
        classifier_layer.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, init_lr=0.1, epoch=epoch_num)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            epoch_num += 1
            iter_source = iter(source_loader)
        if i % len_train_target == 0:
            iter_target = iter(target_loader)
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), Variable(inputs_target), Variable(labels_source)
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        features = base_network(inputs)
        if net_config["use_bottleneck"]:
            features = bottleneck_layer(features)

        outputs = classifier_layer(features)

        classifier_loss = class_criterion(outputs.narrow(0, 0, inputs.size(0)/2), labels_source)
        ## switch between different transfer loss
        if loss_config["name"] == "DANN":
            ad_net.train(True)
            transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, use_gpu)
        elif loss_config["name"] == "JAN":
            softmax_out = softmax_layer(outputs)
            transfer_loss = transfer_criterion([features.narrow(0, 0, features.size(0)/2), softmax_out.narrow(0, 0, softmax_out.size(0)/2)], [features.narrow(0, features.size(0)/2, features.size(0)/2), softmax_out.narrow(0, softmax_out.size(0)/2, softmax_out.size(0)/2)], **loss_config["params"])
        elif loss_config["name"] == "DAN":
            transfer_loss = transfer_criterion(features.narrow(0, 0, features.size(0)/2), features.narrow(0, features.size(0)/2, features.size(0)/2), **loss_config["params"])


        total_loss = loss_config["trade_off"] * transfer_loss + classifier_loss
        total_loss.backward()
        optimizer.step()
    torch.save(best_model, config["output_path"])
    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Transfer Learning')
    parser.add_argument('gpu_id', type=str, nargs='?', default='0', help="device id to run")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id 

    # train config
    
    config = {}
    config["num_iterations"] = 80000
    config["test_interval"] = 500
    config["output_path"] = "../snapshot/best_model.pth.tar"
    config["loss"] = {"name":"DAN", "trade_off":1.0}
    config["data"] = {"list_path":{"source":"../data/cifar/fake.txt", "target":"../data/cifar/real.txt", "test":"../data/cifar/test.txt"}, "batch_size":{"train":128, "test":4} }
    config["network"] = {"name":"ResNet20Cifar", "use_bottleneck":False, "bottleneck_dim":256}
    print(config["loss"])
    print(transfer_classification(config))
    
    # predict config
    '''
    config = {}
    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224}
    config["model_path"] = "../snapshot/best_model.pth.tar"
    config["image_path"] = "/home/caozhangjie/dataset/office/domain_adaptation_images/webcam/images/headphones/frame_0023.jpg"
    print(predict_one_image(config))
    '''
