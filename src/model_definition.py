#!/usr/bin/env python3
# -*- mode: python -*-
# =============================================================================
#  @@-COPYRIGHT-START-@@
#
#  Copyright (c) 2022 of Qualcomm Innovation Center, Inc. All rights reserved.
#
#  @@-COPYRIGHT-END-@@
# =============================================================================

"""Class for downloading and setting up of optmized and original resnet model for AIMET model zoo"""
# pylint:disable = import-error, wrong-import-order
# adding this due to docker image not setup yet

import os
import json
import torch
import torchvision
from aimet_torch.quantsim import QuantizationSimModel, load_encodings_to_sim
from aimet_torch.model_preparer import prepare_model
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from downloader import Downloader
from aimet_torch.batch_norm_fold import fold_all_batch_norms

 
class ResNet(Downloader):
    """ResNet parent class with automated loading of weights and providing a QuantSim with pre-computed encodings"""
    #pylint:disable = unused-argument
    def __init__(self, config=None, device=None,dataloader=None, **kwargs):
        """
        :param model_config:             named model config from which to obtain model artifacts and arguments.
                                         If provided, overwrites the other arguments passed to this object
        """
        self.device = device or torch.device("cuda")
        self.cfg = False
        self.dataloader=dataloader
        if config:
            self.cfg =config
            
        if self.cfg:
            self.input_shape = tuple(
                x if x is not None else 1 for x in self.cfg["input_shape"]
            )
        self.resnet_variant = self.cfg["name"]
        supported_resnet_variants = {"resnet50"}
        if self.resnet_variant not in supported_resnet_variants:
            raise NotImplementedError(
                f"Only support variants in {supported_resnet_variants}"
            )
        self.model = getattr(torchvision.models, self.resnet_variant)(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

    def from_pretrained(self, quantized=False):
        """load pretrained weights"""
        if not self.cfg:
            raise NotImplementedError(
                "There are no pretrained weights available for the model_config passed"
            )
        
        if quantized:
            
            self.model = prepare_model(self.model)
            
            if "cle" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
                equalize_model(self.model, self.input_shape)
                
            if "bn" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
                fold_all_batch_norms(self.model, self.input_shape)
                
            if "adaround" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
                params = AdaroundParameters(data_loader=self.dataloader, num_batches=1,default_num_iterations=1)
                dummy_input = torch.rand(self.input_shape, device=self.device)
                self.model = Adaround.apply_adaround(self.model,
                                                    dummy_input,
                                                    params,
                                                    path=self.cfg['exports_path'],
                                                    filename_prefix='Adaround',
                                                    default_param_bw=self.cfg["optimization_config"]["quantization_configuration"]["param_bw"],
                                                    default_quant_scheme=self.cfg["optimization_config"]["quantization_configuration"]["quant_scheme"])
                
        else:
            self.model = getattr(torchvision.models, self.resnet_variant)(pretrained=True)
            self.model.to(self.device)
        self.model.eval()

    def get_quantsim(self, quantized=False):
        """get quantsim object with pre-loaded encodings"""
        if not self.cfg:
            raise NotImplementedError(
                "There is no Quantization Simulation available for the model_config passed"
            )
        if quantized:
            self.from_pretrained(quantized=True)
        else:
            self.from_pretrained(quantized=False)
        dummy_input = torch.rand(self.input_shape, device=self.device)
        kwargs = {
            "quant_scheme": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["quant_scheme"],
            "default_param_bw": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["param_bw"],
            "default_output_bw": self.cfg["optimization_config"][
                "quantization_configuration"
            ]["output_bw"], 
            "dummy_input": dummy_input,
        }
        sim = QuantizationSimModel(self.model, **kwargs)
        if quantized and "adaround" in self.cfg["optimization_config"]["quantization_configuration"]["techniques"]:
            sim.set_and_freeze_param_encodings(encoding_path=self.cfg['exports_path']+'/Adaround.encodings')
            print("set_and_freeze_param_encodings finished!")
        sim.model.eval()
        return sim
