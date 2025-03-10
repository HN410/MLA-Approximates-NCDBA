# https://github.com/pongkun/Code-for-DODA/tree/main

import torch
import os
from utils.utils import *
from utils.network_arch_resnet import *
from utils.model_units import MyModel, BNLeNet5, BNMLP, MLResBlock, TableMLP, OneVSOneAdjuster, LogitAdjuster
from utils.data_units import MINI_IMAGENET_DATA, IMAGENET_DATA, get_input_dimension

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

PRETRAINED_PATH = "clones/Code-for-DODA/results/cifar100/ce@N_500_ir_100_wonorm/checkpoint.pth.tar"

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class ResNet_s(MyModel):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, logit_adjuster=None, outFeature=False, outAllFeatures=False):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        self.apply(_weights_init)
        if logit_adjuster:
            self.logit_adjuster = logit_adjuster
            
        self.outFeature = outFeature
        self.outAllFeatures = outAllFeatures

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, output_type='feat'):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out1 = out.view(out.size(0), -1)
        
        if self.outFeature:
            if self.outAllFeatures:
                return [out1]
            return out1
        else:
            out = self.forward_last_layer(out1)
            return out
    def belongs_to_fc_layers(cls, name: str):
        # provisional
        return name.startswith("linear")
    @classmethod
    def belongs_to_bn_bias(cls, name: str):
        raise NotImplementedError
    @classmethod
    def belongs_to_bn_weight(cls, name: str):
        raise NotImplementedError
    @classmethod
    def belongs_to_identity_bn(cls, name: str):
        raise NotImplementedError
    
    def get_fc_weight(self):
        return self.linear.weight
    
    def get_fc_bias(self):
        return self.linear.bias
    
    def set_fc_weight(self, tensor: torch.Tensor):
        self.linear.weight.data = tensor
    
    def set_fc_bias(self, tensor: torch.Tensor):
        self.linear.bias.data = tensor
    
    def get_fc_params_list(self):
        return [val for key, val in self.named_parameters() if self.belongs_to_fc_layers(key)]
    
    def forward_last_layer(self, x: torch.Tensor):
        out = self.linear(x)
        if self.logit_adjuster:
            out = self.logit_adjuster(out)
        return out
    def get_features_n(self):
        return 1
    
    def get_last_layer_weights_names_list(self):
        raise NotImplementedError
    def check_valid_wd_target(self, name: str, args: BaseArgs):
        raise NotImplementedError()

def get_doda_model(logit_adjuster: LogitAdjuster):
    num_class = 100
    model = ResNet_s(BasicBlock, [5,5,5], num_class, use_norm=False, logit_adjuster=logit_adjuster)
    state_dict = torch.load(PRETRAINED_PATH, map_location="cpu")['state_dict']
    # linear = state_dict["linear.weight"]
    # # ノルムを正規化
    # linear = linear / torch.norm(linear, dim=1, keepdim=True)
    # state_dict["linear.weight"] = linear
    model.load_state_dict(state_dict)
    return model
