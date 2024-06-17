# -*- coding: utf-8 -*-
import GLOB as glob
import torch
import torch.nn as nn
from thop import profile, clever_format
from models.pose.hourglass.hourglass import build_hourglass as hourglass
from models.pose.hourglass.hourglass_ms import build_hourglass_ms as hourglass_ms




def create_model(args):
    logger = glob.get_value('logger')
    model, info = None, ''
    # region 1.1 model initialize
    if args.arch == 'Hourglass':
        model = hourglass(args.num_classes, args.stack_num).to(args.device)
        info = 'Hourglass (stack: {})'.format(args.stack_num)
    elif args.arch == 'Hourglass_MS':
        model = hourglass_ms(args.stream_num, args.num_classes, args.expend, args.noisy_factor, args.device).to(args.device)
        info = 'Hourglass_MS {} * (stream: {}, stack: {})'.format(args.stream_num, args.stream_num, args.stack_num)
    # endregion

    # region 1.2 FLOPs calculate
    input_shape = (3, 256, 256)
    input_tensor = torch.randn(1, *input_shape).to(args.device)
    mac = calculate_mac(model, input=input_tensor)/1000000
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    logger.print('L1', '=> model: {} | params: {} | FLOPs: {} | MAC: {}'.format(info, params, flops, mac))
    # endregion
    return model


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

















