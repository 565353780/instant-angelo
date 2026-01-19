"""Visualization utilities"""

import torch
import numpy as np


def tensorboard_image(image, from_range=(0, 1), cmap=None):
    """准备图像用于 TensorBoard 显示
    
    Args:
        image: [H, W, C] 或 [H, W] 张量
        from_range: 输入值范围
        cmap: 颜色映射（用于灰度图）
        
    Returns:
        [C, H, W] 准备好的图像
    """
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
        
    # 归一化到 [0, 1]
    image = (image - from_range[0]) / (from_range[1] - from_range[0])
    image = image.clamp(0, 1)
    
    # 处理灰度图
    if image.ndim == 2:
        image = image.unsqueeze(-1).expand(-1, -1, 3)
        
    # 转换为 [C, H, W]
    if image.shape[-1] in [1, 3, 4]:
        image = image.permute(2, 0, 1)
        
    return image


def depth_to_colormap(depth, near=None, far=None):
    """将深度图转换为彩色图
    
    Args:
        depth: [H, W] 深度图
        near: 近平面
        far: 远平面
        
    Returns:
        [H, W, 3] 彩色深度图
    """
    if near is None:
        near = depth.min()
    if far is None:
        far = depth.max()
        
    # 归一化
    depth_normalized = (depth - near) / (far - near + 1e-8)
    depth_normalized = depth_normalized.clamp(0, 1)
    
    # 使用简单的颜色映射 (turbo-like)
    r = depth_normalized
    g = 1 - 2 * (depth_normalized - 0.5).abs()
    b = 1 - depth_normalized
    
    return torch.stack([r, g, b], dim=-1)


def normal_to_colormap(normal):
    """将法向量转换为颜色
    
    Args:
        normal: [H, W, 3] 法向量（已归一化）
        
    Returns:
        [H, W, 3] 彩色法向量图
    """
    return normal * 0.5 + 0.5
