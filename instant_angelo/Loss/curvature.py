"""Curvature Loss for surface smoothness"""

import torch


def curvature_loss(hessian, outside=None):
    """曲率损失: 约束表面平滑
    
    Args:
        hessian: [*, 3] Hessian 对角元素
        outside: [*] 布尔掩码，True 表示在场景外部
        
    Returns:
        loss: 标量损失值
    """
    if hessian is None:
        return torch.tensor(0.0)
        
    laplacian = hessian.sum(dim=-1).abs()
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    
    if outside is not None:
        return (laplacian * (~outside).float()).mean()
    else:
        return laplacian.mean()
