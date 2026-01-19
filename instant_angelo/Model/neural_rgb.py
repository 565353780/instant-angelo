"""Neural RGB Network for color prediction"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def get_spherical_harmonics(levels, dirs):
    """计算球谐函数编码
    
    Args:
        levels: SH 级别 (0-4)
        dirs: [*, 3] 方向向量
        
    Returns:
        [*, (levels+1)^2] SH 编码
    """
    result = []
    x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
    
    # Level 0
    result.append(0.28209479177387814 * torch.ones_like(x))
    
    if levels >= 1:
        # Level 1
        result.append(0.4886025119029199 * y)
        result.append(0.4886025119029199 * z)
        result.append(0.4886025119029199 * x)
        
    if levels >= 2:
        # Level 2
        result.append(1.0925484305920792 * x * y)
        result.append(1.0925484305920792 * y * z)
        result.append(0.31539156525252005 * (3 * z * z - 1))
        result.append(1.0925484305920792 * x * z)
        result.append(0.5462742152960396 * (x * x - y * y))
        
    if levels >= 3:
        # Level 3
        result.append(0.5900435899266435 * y * (3 * x * x - y * y))
        result.append(2.890611442640554 * x * y * z)
        result.append(0.4570457994644658 * y * (5 * z * z - 1))
        result.append(0.3731763325901154 * z * (5 * z * z - 3))
        result.append(0.4570457994644658 * x * (5 * z * z - 1))
        result.append(1.445305721320277 * z * (x * x - y * y))
        result.append(0.5900435899266435 * x * (x * x - 3 * y * y))
        
    if levels >= 4:
        # Level 4
        result.append(2.5033429417967046 * x * y * (x * x - y * y))
        result.append(1.7701307697799304 * y * z * (3 * x * x - y * y))
        result.append(0.9461746957575601 * x * y * (7 * z * z - 1))
        result.append(0.6690465435572892 * y * z * (7 * z * z - 3))
        result.append(0.10578554691520431 * (35 * z * z * z * z - 30 * z * z + 3))
        result.append(0.6690465435572892 * x * z * (7 * z * z - 3))
        result.append(0.47308734787878004 * (x * x - y * y) * (7 * z * z - 1))
        result.append(1.7701307697799304 * x * z * (x * x - 3 * y * y))
        result.append(0.6258357354491761 * (x * x * (x * x - 3 * y * y) - y * y * (3 * x * x - y * y)))
        
    return torch.cat(result, dim=-1)


def positional_encoding(x, levels):
    """Fourier positional encoding
    
    Args:
        x: [*, D] 输入
        levels: 编码级别数
        
    Returns:
        [*, D * 2 * levels] 编码后的特征
    """
    result = []
    for i in range(levels):
        freq = 2.0 ** i
        result.append(torch.sin(x * freq * np.pi))
        result.append(torch.cos(x * freq * np.pi))
    return torch.cat(result, dim=-1)


class MLPwithSkipConnection(nn.Module):
    """带 skip connection 的 MLP"""
    
    def __init__(self, layer_dims, skip_connection=[], use_weightnorm=False):
        super().__init__()
        self.skip_connection = skip_connection
        self.linears = nn.ModuleList()
        
        layer_dim_pairs = list(zip(layer_dims[:-1], layer_dims[1:]))
        for li, (k_in, k_out) in enumerate(layer_dim_pairs):
            if li in self.skip_connection:
                k_in += layer_dims[0]
            linear = nn.Linear(k_in, k_out)
            if use_weightnorm:
                linear = nn.utils.parametrizations.weight_norm(linear)
            self.linears.append(linear)
            
        self.activ = nn.ReLU(inplace=True)
        
    def forward(self, x):
        input_x = x
        for li, linear in enumerate(self.linears):
            if li in self.skip_connection:
                x = torch.cat([x, input_x], dim=-1)
            x = linear(x)
            if li < len(self.linears) - 1:
                x = self.activ(x)
        return x


class NeuralRGB(nn.Module):
    """Neural RGB Network
    
    基于 IDR 风格的颜色预测网络
    """
    
    def __init__(self, cfg_rgb, feat_dim, appear_embed=None):
        super().__init__()
        self.cfg_rgb = cfg_rgb
        self.mode = cfg_rgb.mode
        
        # 构建视角编码器
        encoding_view_dim = self._build_encoding(cfg_rgb.encoding_view)
        
        # 输入维度: 点坐标 + 法向量 + 视角编码 + 特征 + 外观嵌入
        if cfg_rgb.mode == "idr":
            input_base_dim = 6  # point + normal
        else:
            input_base_dim = 3  # just point
            
        appear_dim = appear_embed.dim if appear_embed is not None and appear_embed.enabled else 0
        input_dim = input_base_dim + encoding_view_dim + feat_dim + appear_dim
        
        # 构建 MLP
        layer_dims = [input_dim] + [cfg_rgb.mlp.hidden_dim] * cfg_rgb.mlp.num_layers + [3]
        self.mlp = MLPwithSkipConnection(
            layer_dims,
            skip_connection=cfg_rgb.mlp.skip,
            use_weightnorm=cfg_rgb.mlp.weight_norm,
        )
        
    def _build_encoding(self, cfg_encoding_view):
        """构建视角编码器"""
        if cfg_encoding_view.type == "fourier":
            self.encode_view = lambda x: positional_encoding(x, cfg_encoding_view.levels)
            encoding_view_dim = 3 * 2 * cfg_encoding_view.levels
        elif cfg_encoding_view.type == "spherical":
            self.encode_view = lambda x: get_spherical_harmonics(cfg_encoding_view.levels, x)
            encoding_view_dim = (cfg_encoding_view.levels + 1) ** 2
        else:
            raise NotImplementedError(f"Unknown encoding type: {cfg_encoding_view.type}")
        return encoding_view_dim
        
    def forward(self, points, normals, rays_unit, feats, app=None):
        """
        Args:
            points: [*, 3] 3D 点
            normals: [*, 3] 法向量
            rays_unit: [*, 3] 视角方向（单位向量）
            feats: [*, D] SDF 网络输出的特征
            app: [*, A] 外观嵌入（可选）
            
        Returns:
            rgb: [*, 3] RGB 颜色
        """
        view_enc = self.encode_view(rays_unit)
        
        if self.mode == "idr":
            # IDR 风格: 使用点坐标和法向量
            inputs = [points, normals, view_enc, feats]
        else:
            # NeRF 风格: 只使用点坐标
            inputs = [points, view_enc, feats]
            
        if app is not None:
            inputs.append(app)
            
        x = torch.cat(inputs, dim=-1)
        rgb = self.mlp(x)
        rgb = torch.sigmoid(rgb)
        
        return rgb
