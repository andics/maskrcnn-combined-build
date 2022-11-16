import torch
import numpy as np

def check_layer_freeze_pattern(model_base_prefix, iteration, cfg, model, optimizer):
    if cfg.SOLVER.LAYER_FREEZE_PATTERN[0] is not "Nothing":
        for freeze_layer_array in cfg.SOLVER.LAYER_FREEZE_PATTERN:
            if iteration >= freeze_layer_array[0] and iteration < freeze_layer_array[1]:
                #print("Attempting to freeze layers: ", iteration)
                [freeze_layer_if_unfrozen(model_base_prefix, layer, cfg, model, optimizer) for layer in freeze_layer_array[2:]]
            else:
                #print("Attempting to unfreeze layers: ", iteration)
                [unfreeze_layer_if_frozen(model_base_prefix, layer, cfg, model, optimizer) for layer in freeze_layer_array[2:]]


def freeze_layer_if_unfrozen(model_base_prefix, layer, cfg, model, optimizer):
    for parameter in eval(model_base_prefix + layer + ".parameters()"):
        parameter.requires_grad = False
        #parameter.grad.zero_()
        #parameter.grad = torch.zeros(parameter.shape, dtype=torch.float32)
        #parameter._grad = torch.zeros(parameter.shape, dtype=torch.float32)
        #parameter.grad = 0


def unfreeze_layer_if_frozen(model_base_prefix, layer, cfg, model, optimizer):
    for parameter in eval(model_base_prefix + layer + ".parameters()"):
        parameter.requires_grad = True