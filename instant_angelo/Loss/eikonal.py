"""Eikonal Loss for SDF regularization"""

import torch


def eikonal_loss(gradients, outside=None):
    """Eikonal 损失: 约束 SDF 梯度范数为 1
    
    Args:
        gradients: [*, 3] SDF 梯度
        outside: [*] 布尔掩码，True 表示在场景外部
        
    Returns:
        loss: 标量损失值
    """
    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    
    if outside is not None:
        return (gradient_error * (~outside).float()).mean()
    else:
        return gradient_error.mean()
