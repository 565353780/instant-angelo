"""Data utilities"""

import torch
import numpy as np


def normalize_poses(poses, center=None, scale=None):
    """归一化相机姿态
    
    Args:
        poses: [N, 3, 4] 相机姿态
        center: 场景中心
        scale: 场景缩放
        
    Returns:
        normalized_poses: 归一化后的姿态
        center: 使用的中心
        scale: 使用的缩放
    """
    if center is None:
        center = poses[:, :3, 3].mean(0)
    if scale is None:
        scale = (poses[:, :3, 3] - center).norm(dim=-1).mean()
        
    normalized_poses = poses.clone()
    normalized_poses[:, :3, 3] = (poses[:, :3, 3] - center) / scale
    
    return normalized_poses, center, scale


def create_spiral_poses(poses, n_frames=120, n_rots=2, zrate=0.5):
    """创建螺旋相机轨迹
    
    Args:
        poses: [N, 3, 4] 参考姿态
        n_frames: 帧数
        n_rots: 旋转圈数
        zrate: Z 轴变化率
        
    Returns:
        spiral_poses: [n_frames, 3, 4] 螺旋轨迹
    """
    # 计算平均位置和方向
    center = poses[:, :3, 3].mean(0)
    up = poses[:, :3, 1].mean(0)
    up = up / up.norm()
    
    # 计算半径
    radius = (poses[:, :3, 3] - center).norm(dim=-1).mean()
    
    # 生成螺旋轨迹
    spiral_poses = []
    for i in range(n_frames):
        theta = 2 * np.pi * n_rots * i / n_frames
        
        # 相机位置
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = radius * zrate * np.sin(2 * np.pi * i / n_frames)
        pos = center + torch.tensor([x, y, z], dtype=poses.dtype)
        
        # 相机朝向
        forward = center - pos
        forward = forward / forward.norm()
        
        # 相机右向
        right = torch.cross(forward, up)
        right = right / right.norm()
        
        # 相机上向
        new_up = torch.cross(right, forward)
        
        # 构建姿态矩阵
        pose = torch.stack([right, new_up, -forward, pos], dim=1)
        spiral_poses.append(pose)
        
    return torch.stack(spiral_poses, dim=0)
