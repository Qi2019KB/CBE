# -*- coding: utf-8 -*-
import GLOB as glob
import torch
import torch.nn as nn
from thop import profile, clever_format
from models.classification.wideresnet import build_wideresnet as wideresnet
from models.classification.wideresnet_ms import build_wideresnet_ms as wideresnet_ms


def create_model(args):
    args = set_parameters(args)
    logger = glob.get_value('logger')
    model, info = None, ''
    # region 1.1 model initialize
    if args.arch == 'WideResNet':
        model = wideresnet(args.model_depth, args.model_width, 0, args.num_classes).to(args.device)
        info = 'WideResNet {}x{}'.format(args.model_depth, args.model_width)
    elif args.arch == 'WideResNet_MS':
        model = wideresnet_ms(args.model_depth, args.model_width, 0, args.num_classes, args.stream_num, args.noisy_factor, args.device).to(args.device)
        info = 'WideResNet_MS {} * {}x{}'.format(args.stream_num, args.model_depth, args.model_width)
    # endregion

    # region 1.2 FLOPs calculate
    input_shape = (3, 32, 32)
    input_tensor = torch.randn(1, *input_shape).to(args.device)
    mac = calculate_mac(model, input=input_tensor)/1000000
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    logger.print('L1', '=> model: {} | params: {} | FLOPs: {} | MAC: {}'.format(info, params, flops, mac))
    # endregion
    return model


def set_parameters(args):
    if args.dataset == 'CIFAR10':
        if args.arch in ['WideResNet', 'WideResNet_MS']:
            args.model_depth = 28
            args.model_width = 2
        elif args.arch in ['ResNeXt', 'ResNeXt_MS']:
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4
    elif args.dataset == 'CIFAR100':
        if args.arch in ['WideResNet', 'WideResNet_MS']:
            args.model_depth = 28
            args.model_width = 8
        elif args.arch in ['ResNeXt', 'ResNeXt_MS']:
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    return args


# 计算MAC
def calculate_mac(model, input):
    # 设置模型为eval模式
    model.eval()
    macs = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # 计算卷积层的MAC
            macs += layer.weight.numel() * input.numel()
        elif isinstance(layer, nn.Linear):
            # 计算全连接层的MAC
            macs += layer.weight.numel() * input.numel()
    return macs