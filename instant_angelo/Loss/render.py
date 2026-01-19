import torch
import torch.nn.functional as F


def render_loss(pred_rgb, gt_rgb, loss_type='l1'):
    """渲染损失
    
    Args:
        pred_rgb: [*, 3] 预测 RGB
        gt_rgb: [*, 3] 真实 RGB
        loss_type: 损失类型 ('l1', 'l2', 'smooth_l1')
        
    Returns:
        loss: 标量损失值
    """
    if loss_type == 'l1':
        return F.l1_loss(pred_rgb, gt_rgb)
    elif loss_type == 'l2':
        return F.mse_loss(pred_rgb, gt_rgb)
    elif loss_type == 'smooth_l1':
        return F.smooth_l1_loss(pred_rgb, gt_rgb)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def mask_loss(pred_opacity, gt_mask):
    """掩码损失
    
    Args:
        pred_opacity: [*] 预测不透明度
        gt_mask: [*] 真实掩码
        
    Returns:
        loss: 标量损失值
    """
    pred_opacity = pred_opacity.clamp(1e-3, 1 - 1e-3)
    return F.binary_cross_entropy(pred_opacity, gt_mask.float())


def sparsity_loss(sdf_samples, scale=1.0):
    """稀疏性损失: 鼓励 SDF 在远离表面时快速衰减
    
    Args:
        sdf_samples: [*] SDF 采样值
        scale: 缩放因子
        
    Returns:
        loss: 标量损失值
    """
    return torch.exp(-scale * sdf_samples.abs()).mean()


def distortion_loss(weights, points, intervals, ray_indices):
    """Distortion 损失 (来自 MipNeRF-360)
    
    鼓励权重集中在表面附近
    """
    try:
        from torch_efficient_distloss import flatten_eff_distloss
        return flatten_eff_distloss(weights, points, intervals, ray_indices)
    except ImportError:
        # Fallback: 简化版本
        return torch.tensor(0.0, device=weights.device)
