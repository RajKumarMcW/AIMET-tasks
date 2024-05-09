#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================
""" module for evaluation function and forward pass of dataloader"""
import torch
from tqdm import tqdm


def eval_func(model, dataloader):
    """Evaluates the model on validation dataset and returns the classification accuracy"""
    # Get Dataloader
    model.eval()
    correct,corrects = 0,0
    total_samples = 0
    on_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)
            _, prediction = torch.max(output, 1)
            _, predictions = torch.topk(output, k=5, dim=1)
            correct += (prediction == label).sum()
            corrects += torch.sum(predictions==label.view(-1,1)).item() 
            total_samples += len(output)
            
    del dataloader
    top1=float(100 * correct / total_samples)
    top5=float(100 * corrects / total_samples)
    return top1,top5


def forward_pass(model, dataloader):
    """forward pass through the calibration dataset"""
    #pylint:disable = unused-variable
    model.eval()
    on_cuda = next(model.parameters()).is_cuda
    with torch.no_grad():
        for data, label in tqdm(dataloader):
            if on_cuda:
                data, label = data.cuda(), label.cuda()
            output = model(data)
    del dataloader
