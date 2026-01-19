"""Neural SDF Network with Hash Grid Encoding

使用 tiny-cuda-nn 的 HashGrid 编码和小型 MLP 实现高效的 SDF 网络
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("Warning: tinycudann not available, using vanilla implementation")


class VanillaMLP(nn.Module):
    """Vanilla MLP with geometric initialization for SDF"""
    
    def __init__(self, dim_in, dim_out, hidden_dim, num_layers, 
                 skip=[], geometric_init=True, weight_norm=True,
                 out_bias=0.5, inside_out=False, activ_beta=100.0):
        super().__init__()
        self.skip = skip
        self.num_layers = num_layers
        
        layers = []
        for i in range(num_layers + 1):
            if i == 0:
                in_dim = dim_in
            elif i in skip:
                in_dim = hidden_dim + dim_in
            else:
                in_dim = hidden_dim
                
            out_dim = dim_out if i == num_layers else hidden_dim
            
            layer = nn.Linear(in_dim, out_dim)
            
            if geometric_init:
                self._geometric_init(layer, i, num_layers, out_bias, inside_out, hidden_dim, dim_in)
                
            if weight_norm:
                layer = nn.utils.parametrizations.weight_norm(layer)
                
            layers.append(layer)
            
        self.layers = nn.ModuleList(layers)
        self.activ = nn.Softplus(beta=activ_beta)
        
    def _geometric_init(self, layer, layer_idx, num_layers, out_bias, inside_out, hidden_dim, dim_in):
        """Geometric initialization for SDF network"""
        if layer_idx == num_layers:
            # Last layer
            nn.init.normal_(layer.weight, mean=math.sqrt(math.pi) / math.sqrt(hidden_dim), std=0.0001)
            nn.init.constant_(layer.bias, -out_bias if not inside_out else out_bias)
        elif layer_idx == 0:
            # First layer
            nn.init.constant_(layer.bias, 0.0)
            nn.init.constant_(layer.weight[:, 3:], 0.0)
            nn.init.normal_(layer.weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(hidden_dim))
        else:
            # Hidden layers
            nn.init.constant_(layer.bias, 0.0)
            nn.init.normal_(layer.weight, 0.0, math.sqrt(2) / math.sqrt(hidden_dim))
            
    def forward(self, x, with_sdf=True, with_feat=True):
        input_x = x
        for i, layer in enumerate(self.layers):
            if i in self.skip:
                x = torch.cat([x, input_x], dim=-1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.activ(x)
                
        if with_sdf and with_feat:
            return x[..., :1], x
        elif with_sdf:
            return x[..., :1], None
        else:
            return None, x


class ProgressiveBandHashGrid(nn.Module):
    """Progressive Hash Grid Encoding with coarse-to-fine training"""
    
    def __init__(self, config):
        super().__init__()
        self.n_input_dims = 3
        
        # HashGrid 参数
        self.n_levels = config.levels
        self.n_features_per_level = config.hashgrid.dim
        self.log2_hashmap_size = config.hashgrid.dict_size
        self.base_resolution = 2 ** config.hashgrid.min_logres
        self.max_resolution = 2 ** config.hashgrid.max_logres
        self.per_level_scale = np.exp((np.log(self.max_resolution) - np.log(self.base_resolution)) / (self.n_levels - 1))
        
        # Coarse-to-fine 参数
        self.c2f_enabled = config.coarse2fine.enabled
        self.init_active_level = config.coarse2fine.init_active_level
        self.c2f_step = config.coarse2fine.step
        self.c2f_start_step = config.coarse2fine.start_step
        self.current_level = self.init_active_level
        
        # 计算每个级别的分辨率
        self.resolutions = []
        for lv in range(self.n_levels):
            size = int(np.floor(self.base_resolution * self.per_level_scale ** lv)) + 1
            self.resolutions.append(size)
        
        self.n_output_dims = self.n_levels * self.n_features_per_level
        
        # 创建 TCNN HashGrid
        if TCNN_AVAILABLE:
            encoding_config = {
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_resolution,
                "per_level_scale": self.per_level_scale,
            }
            self.encoding = tcnn.Encoding(3, encoding_config)
        else:
            # Fallback to frequency encoding
            self.encoding = None
            self.n_output_dims = 3 * 2 * 10  # Fallback
            
        # 初始化 mask
        self.register_buffer('mask', torch.zeros(self.n_output_dims))
        self.mask[:self.current_level * self.n_features_per_level] = 1.0
        
    def forward(self, x):
        """
        Args:
            x: [*, 3] 归一化到 [0, 1] 的坐标
        Returns:
            [*, n_output_dims] 编码后的特征
        """
        shape = x.shape[:-1]
        x_flat = x.view(-1, 3)
        
        if TCNN_AVAILABLE and self.encoding is not None:
            enc = self.encoding(x_flat)
            enc = enc * self.mask
        else:
            # Fallback: simple frequency encoding
            freqs = 2.0 ** torch.arange(10, device=x.device)
            enc = torch.cat([torch.sin(x_flat @ freqs.view(1, -1).expand(3, -1).T.flatten().view(3, -1)),
                            torch.cos(x_flat @ freqs.view(1, -1).expand(3, -1).T.flatten().view(3, -1))], dim=-1)
            
        return enc.view(*shape, -1)
    
    def update_step(self, global_step):
        """更新 coarse-to-fine 级别"""
        if not self.c2f_enabled:
            self.current_level = self.n_levels
        else:
            if global_step < self.c2f_start_step:
                self.current_level = self.init_active_level
            else:
                self.current_level = min(
                    self.init_active_level + (global_step - self.c2f_start_step) // self.c2f_step,
                    self.n_levels
                )
        self.mask[:self.current_level * self.n_features_per_level] = 1.0
        return self.current_level


class NeuralSDF(nn.Module):
    """Neural SDF Network
    
    结合 HashGrid 编码和小型 MLP 的高效 SDF 网络
    """
    
    def __init__(self, cfg_sdf):
        super().__init__()
        self.cfg_sdf = cfg_sdf
        self.radius = cfg_sdf.encoding.hashgrid.range[1]
        
        # 构建编码器
        self.encoding = ProgressiveBandHashGrid(cfg_sdf.encoding)
        
        # 构建 MLP
        input_dim = 3 + self.encoding.n_output_dims  # xyz + encoding
        self.feature_dim = cfg_sdf.feature_dim
        
        self.mlp = VanillaMLP(
            dim_in=input_dim,
            dim_out=self.feature_dim,
            hidden_dim=cfg_sdf.mlp.hidden_dim,
            num_layers=cfg_sdf.mlp.num_layers,
            skip=cfg_sdf.mlp.skip,
            geometric_init=cfg_sdf.mlp.geometric_init,
            weight_norm=cfg_sdf.mlp.weight_norm,
            out_bias=cfg_sdf.mlp.out_bias,
            inside_out=cfg_sdf.mlp.inside_out,
            activ_beta=cfg_sdf.mlp.activ_params.beta,
        )
        
        # 梯度计算参数
        self.grad_type = cfg_sdf.gradient.type
        self.grad_taps = cfg_sdf.gradient.taps
        self._finite_difference_eps = None
        
        # Coarse-to-fine 相关
        self.warm_up_end = 0
        self.active_levels = cfg_sdf.encoding.coarse2fine.init_active_level
        self.normal_eps = 1.0 / self.encoding.resolutions[self.active_levels - 1]
        
    def forward(self, points, with_sdf=True, with_feat=True):
        """
        Args:
            points: [*, 3] 世界坐标
            with_sdf: 是否返回 SDF
            with_feat: 是否返回特征
        Returns:
            sdf: [*, 1]
            feat: [*, feature_dim]
        """
        points_enc = self.encode(points)
        sdf, feat = self.mlp(points_enc, with_sdf=with_sdf, with_feat=with_feat)
        return sdf, feat
    
    def sdf(self, points):
        """仅计算 SDF 值"""
        sdf, _ = self.forward(points, with_sdf=True, with_feat=False)
        return sdf
    
    def encode(self, points):
        """编码 3D 点"""
        # 归一化到 [0, 1]
        vol_min, vol_max = self.cfg_sdf.encoding.hashgrid.range
        points_normalized = (points - vol_min) / (vol_max - vol_min)
        points_normalized = points_normalized.clamp(0, 1)
        
        # HashGrid 编码
        points_enc = self.encoding(points_normalized)
        
        # 拼接原始坐标
        points_enc = torch.cat([points, points_enc], dim=-1)
        return points_enc
    
    def compute_gradients(self, x, training=False, sdf=None):
        """计算 SDF 梯度和 Hessian
        
        Args:
            x: [*, 3] 采样点
            training: 是否训练模式
            sdf: 已计算的 SDF 值（可选）
            
        Returns:
            gradient: [*, 3]
            hessian: [*, 3] (对角元素) 或 None
        """
        if self.grad_type == "analytic":
            return self._compute_gradients_analytic(x, training, sdf)
        else:
            return self._compute_gradients_finite_diff(x, training, sdf)
    
    def _compute_gradients_analytic(self, x, training=False, sdf=None):
        """使用自动微分计算梯度"""
        # 使用 torch.enable_grad() 确保即使在 no_grad 上下文中也能计算梯度
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            sdf_val = self.sdf(x)
            
            gradient = torch.autograd.grad(
                sdf_val, x,
                grad_outputs=torch.ones_like(sdf_val),
                create_graph=training,
                retain_graph=training,
                only_inputs=True
            )[0]
        
        # 暂不计算 Hessian（解析法计算 Hessian 开销较大）
        hessian = None
        
        return gradient, hessian
    
    def _compute_gradients_finite_diff(self, x, training=False, sdf=None):
        """使用有限差分计算梯度"""
        eps = self.normal_eps
        
        if self.grad_taps == 6:
            # 6-tap finite difference
            eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)
            eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)
            eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)
            
            sdf_x_pos = self.sdf(x + eps_x)
            sdf_x_neg = self.sdf(x - eps_x)
            sdf_y_pos = self.sdf(x + eps_y)
            sdf_y_neg = self.sdf(x - eps_y)
            sdf_z_pos = self.sdf(x + eps_z)
            sdf_z_neg = self.sdf(x - eps_z)
            
            gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
            gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
            gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
            gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)
            
            # Hessian 对角元素
            if training and sdf is not None:
                hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)
                hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)
                hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)
                hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)
            else:
                hessian = None
                
        else:  # 4-tap
            eps = self.normal_eps / np.sqrt(3)
            k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)
            k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)
            k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)
            k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)
            
            sdf1 = self.sdf(x + k1 * eps)
            sdf2 = self.sdf(x + k2 * eps)
            sdf3 = self.sdf(x + k3 * eps)
            sdf4 = self.sdf(x + k4 * eps)
            
            gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps)
            
            if training and sdf is not None:
                hessian_trace = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / eps ** 2
                hessian = torch.cat([hessian_trace, hessian_trace, hessian_trace], dim=-1) / 3.0
            else:
                hessian = None
                
        return gradient, hessian
    
    def set_active_levels(self, current_iter=None):
        """更新 coarse-to-fine 级别"""
        self.active_levels = self.encoding.update_step(current_iter)
        
    def set_normal_epsilon(self):
        """更新有限差分 epsilon"""
        epsilon_res = self.encoding.resolutions[self.active_levels - 1]
        self.normal_eps = 1. / epsilon_res
        
    def update_step(self, epoch, global_step):
        """训练步骤更新"""
        self.set_active_levels(global_step)
        self.set_normal_epsilon()
