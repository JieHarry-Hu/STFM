#!/usr/bin/env python3
# modified from https://github.com/facebookresearch/SlowFast
"""Model construction functions."""

import torch
from fvcore.common.registry import Registry

import slowfast.utils.logging as logging

logger = logging.get_logger(__name__)

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """
Registry for video model.

The registered object will be called with `obj(cfg)`.
The call should return a `torch.nn.Module` object.
"""


def build_model(cfg, gpu_id=None):
    """
    Builds the video model.
    Args:
        cfg (configs): configs that contains the hyper-parameters to build the
        backbone. Details can be seen in slowfast/config/defaults.py.
        gpu_id (Optional[int]): specify the gpu index to build model.
    """
    if torch.cuda.is_available():
        assert (
            cfg.NUM_GPUS <= torch.cuda.device_count()
        ), "Cannot use more GPU devices than available"
    else:
        assert (
            cfg.NUM_GPUS == 0
        ), "Cuda is not available. Please set `NUM_GPUS: 0 for running on CPUs."

    # Construct the model
    name = cfg.MODEL.MODEL_NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    #read model parameters and freeze some layers
    if cfg.MODEL.ARCH in ['uniformer']:
        checkpoint, freeze_layers = model.get_pretrained_model(cfg)
        if checkpoint:
            logger.info('load pretrained model')
            model.load_state_dict(checkpoint, strict=False)
        if freeze_layers and cfg.MODEL.FINETUNNING:
            logger.info('freeze part layers')
            for n,p in model.named_parameters():
                if n in freeze_layers:
                    p.requires_grad = False
                #logger.info(n,'requires_grad:',p.requires_grad)

    if cfg.NUM_GPUS:
        if gpu_id is None:
            # Determine the GPU used by the current process
            cur_device = torch.cuda.current_device()
        else:
            cur_device = gpu_id
        # Transfer the model to the current GPU device
        model = model.cuda(device=cur_device)
    # Use multi-process data parallel model in the multi-gpu setting
    if cfg.NUM_GPUS > 1:
        # Make model replica operate on the current device
        model = torch.nn.parallel.DistributedDataParallel(
            module=model, device_ids=[cur_device], output_device=cur_device,
            find_unused_parameters=False
        )
    return model
