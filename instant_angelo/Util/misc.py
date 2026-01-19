"""Miscellaneous utilities"""

import os
import random
import numpy as np
import torch


def to_cuda(data, device='cuda'):
    """将数据移动到 GPU"""
    if isinstance(data, dict):
        return {k: to_cuda(v, device) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(to_cuda(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data


def to_cpu(state_dict):
    """将状态字典移动到 CPU"""
    if isinstance(state_dict, dict):
        return {k: to_cpu(v) for k, v in state_dict.items()}
    elif isinstance(state_dict, torch.Tensor):
        return state_dict.cpu()
    else:
        return state_dict


def seed_everything(seed):
    """设置所有随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def get_rank():
    """获取分布式训练的 rank"""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def requires_grad(model, flag=True):
    """设置模型参数是否需要梯度"""
    for p in model.parameters():
        p.requires_grad = flag
