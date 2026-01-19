"""Background NeRF for unbounded scenes"""

import torch
import torch.nn as nn
import numpy as np

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False


class BackgroundNeRF(nn.Module):
    """Background NeRF Network
    
    用于处理无界场景的背景建模
    """
    
    def __init__(self, cfg_bg, appear_embed=None):
        super().__init__()
        self.cfg_bg = cfg_bg
        
        # 构建编码器
        self.n_input_dims = 3
        encoding_dim = self._build_encoding(cfg_bg.encoding)
        
        # 外观嵌入
        appear_dim = appear_embed.dim if appear_embed is not None and appear_embed.enabled else 0
        
        # 构建 MLP
        input_dim = encoding_dim + 3 + appear_dim  # encoding + view dir + appear
        self.mlp = self._build_mlp(input_dim, cfg_bg.mlp)
        
    def _build_encoding(self, cfg_encoding):
        """构建 HashGrid 编码器"""
        n_levels = cfg_encoding.levels
        n_features_per_level = cfg_encoding.hashgrid.dim
        log2_hashmap_size = cfg_encoding.hashgrid.dict_size
        
        if TCNN_AVAILABLE:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 2.0,
            }
            self.encoding = tcnn.Encoding(3, encoding_config)
            encoding_dim = n_levels * n_features_per_level
        else:
            self.encoding = None
            encoding_dim = 3 * 2 * 10  # Fallback frequency encoding
            
        return encoding_dim
    
    def _build_mlp(self, input_dim, cfg_mlp):
        """构建 MLP"""
        layers = []
        in_dim = input_dim
        
        for _ in range(cfg_mlp.num_layers):
            layers.append(nn.Linear(in_dim, cfg_mlp.hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = cfg_mlp.hidden_dim
            
        # 输出层: density + rgb
        layers.append(nn.Linear(in_dim, 4))
        
        return nn.Sequential(*layers)
    
    def _contract_to_sphere(self, x):
        """将无界坐标收缩到单位球内"""
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag > 1
        x_contracted = x.clone()
        x_contracted[mask.expand_as(x)] = (2 - 1 / mag[mask]) * (x[mask.expand_as(x)] / mag[mask.expand_as(x)])
        return x_contracted
    
    def forward(self, points, rays_unit, app=None):
        """
        Args:
            points: [*, 3] 3D 点
            rays_unit: [*, 3] 视角方向
            app: [*, A] 外观嵌入（可选）
            
        Returns:
            rgb: [*, 3] RGB 颜色
            density: [*, 1] 密度
        """
        # 收缩到单位球
        points_contracted = self._contract_to_sphere(points)
        
        # 归一化到 [0, 1]
        points_normalized = points_contracted / 4 + 0.5
        points_normalized = points_normalized.clamp(0, 1)
        
        # 编码
        shape = points.shape[:-1]
        points_flat = points_normalized.view(-1, 3)
        
        if TCNN_AVAILABLE and self.encoding is not None:
            enc = self.encoding(points_flat)
        else:
            # Fallback: frequency encoding
            freqs = 2.0 ** torch.arange(10, device=points.device)
            enc = torch.cat([
                torch.sin(points_flat @ freqs.view(1, -1).expand(3, -1).T.reshape(3, -1)),
                torch.cos(points_flat @ freqs.view(1, -1).expand(3, -1).T.reshape(3, -1))
            ], dim=-1)
            
        enc = enc.view(*shape, -1)
        
        # 拼接输入
        rays_flat = rays_unit.view(*shape, 3)
        inputs = [enc, rays_flat]
        if app is not None:
            inputs.append(app)
        x = torch.cat(inputs, dim=-1)
        
        # MLP
        out = self.mlp(x.view(-1, x.shape[-1]))
        out = out.view(*shape, 4)
        
        rgb = torch.sigmoid(out[..., :3])
        density = F.softplus(out[..., 3:4] - 1)  # shift for better init
        
        return rgb, density.squeeze(-1)


# Import F for softplus
import torch.nn.functional as F
