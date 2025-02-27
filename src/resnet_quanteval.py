#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
''' AIMET Quantsim evaluation code for quantized classification models - Resnet18, Resnet50 '''

import argparse
# from common.utils.image_net_data_loader import ImageNetDataLoader
from utils.image_net_data_loader import ImageNetDataLoader
from dataloaders_and_eval_func import eval_func, forward_pass
from model_definition import ResNet
import torch
import os
import json
import aimet_torch


def arguments(raw_args):
    """ Parse input arguments """
    parser = argparse.ArgumentParser(description='script for classification model quantization')
    parser.add_argument('--config', help='model configuration to use', type=str, required=True)
    args = parser.parse_args(raw_args)
    print(vars(args))
    return args


def main(raw_args=None):
    """ Run evaluations """
    args = arguments(raw_args)
    # Dataloaders
    if os.path.exists(args.config):
        with open(args.config) as f_in:
            config = json.load(f_in)
    encoding_dataloader = ImageNetDataLoader(config['images_dir'],image_size=224,num_samples_per_class=config['calib_per_class']).data_loader
    eval_dataloader = ImageNetDataLoader(config['images_dir'],image_size=224,num_samples_per_class=config['val_per_class']).data_loader

    device = torch.device('cuda' if config['use_cuda'] else 'cpu')
    # Models
    model = ResNet(config = config, device = device,dataloader=encoding_dataloader)
    model.from_pretrained(quantized=False)

    # Evaluate original
    print("\nBaseline:\n")
    top1,top5 = eval_func(model = model.model, dataloader = eval_dataloader)
    print(f'\nFP32 accuracy: Top1: {top1:0.3f}% Top5: {top5:0.3f}%')
    
    print("\nPTQ")
    sim = model.get_quantsim(quantized=True)
    # Evaluate optimized
    print("\nCompute Encoding")
    sim.compute_encodings(forward_pass, forward_pass_callback_args=encoding_dataloader)
    
    qtop1,qtop5 = eval_func(model = sim.model.to(device), dataloader = eval_dataloader)
    print(f'\nQuantized quantized accuracy: Top1: {qtop1:0.3f}% Top5: {qtop5:0.3f}%')
    
    print("\nExporting Model")
    input_shape = tuple(x if x is not None else 1 for x in config["input_shape"])
    sim.export(path=config['exports_path'], filename_prefix=config['exports_name'], dummy_input=torch.rand(input_shape),onnx_export_args=(aimet_torch.onnx_utils.OnnxExportApiArgs (opset_version=11)))

if __name__ == '__main__':
    main()
