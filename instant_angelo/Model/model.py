"""NeuS Model with nerfacc acceleration

主模型类，集成 SDF 网络、RGB 网络和 nerfacc 加速
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from .neural_sdf import NeuralSDF
from .neural_rgb import NeuralRGB
from .background_nerf import BackgroundNeRF

# 尝试导入最新版 nerfacc
try:
    import nerfacc
    from nerfacc import OccGridEstimator, render_weight_from_alpha, accumulate_along_rays
    NERFACC_AVAILABLE = True
    NERFACC_VERSION = getattr(nerfacc, '__version__', '0.5.0')
except ImportError:
    NERFACC_AVAILABLE = False
    NERFACC_VERSION = None
    print("Warning: nerfacc not available, using vanilla ray marching")


class NeuSModel(nn.Module):
    """NeuS Model with nerfacc acceleration
    
    Features:
    - HashGrid encoding for fast SDF queries
    - OccupancyGrid pruning for efficient ray marching
    - NeuS-style alpha computation for SDF rendering
    """
    
    def __init__(self, cfg_model, cfg_data=None):
        super().__init__()
        self.cfg = cfg_model
        self.radius = cfg_model.radius
        
        # 训练进度（用于 NeuS 退火策略）
        self.progress = 0.0
        
        # 构建模型
        self._build_model(cfg_model, cfg_data)
        
        # 构建 nerfacc 加速结构
        if NERFACC_AVAILABLE and hasattr(cfg_model, 'nerfacc') and cfg_model.nerfacc.enabled:
            self._build_nerfacc(cfg_model)
        else:
            self.estimator = None
            self.estimator_bg = None
            
        # 渲染参数
        self.render_step_size = 1.732 * 2 * self.radius / cfg_model.render.num_samples_per_ray
        self.anneal_end = cfg_model.object.s_var.anneal_end
        
    def _build_model(self, cfg_model, cfg_data):
        """构建网络模块"""
        # SDF 网络
        self.neural_sdf = NeuralSDF(cfg_model.object.sdf)
        
        # RGB 网络
        feat_dim = cfg_model.object.sdf.feature_dim
        appear_embed = cfg_model.appear_embed if hasattr(cfg_model, 'appear_embed') else None
        self.neural_rgb = NeuralRGB(cfg_model.object.rgb, feat_dim, appear_embed)
        
        # 背景网络
        if cfg_model.background.enabled:
            self.background_nerf = BackgroundNeRF(cfg_model.background, appear_embed)
        else:
            self.background_nerf = None
            
        # s-variance 参数
        self.s_var = nn.Parameter(torch.tensor(cfg_model.object.s_var.init_val, dtype=torch.float32))
        
        # 外观嵌入
        if hasattr(cfg_model, 'appear_embed') and cfg_model.appear_embed.enabled:
            num_images = cfg_data.num_images if cfg_data is not None else 100
            self.appear_embed = nn.Embedding(num_images, cfg_model.appear_embed.dim)
            if cfg_model.background.enabled:
                self.appear_embed_outside = nn.Embedding(num_images, cfg_model.appear_embed.dim)
            else:
                self.appear_embed_outside = None
        else:
            self.appear_embed = None
            self.appear_embed_outside = None
            
        # 白色背景标志
        self.white_background = cfg_model.background.white if hasattr(cfg_model.background, 'white') else False
        
    def _build_nerfacc(self, cfg_model):
        """构建 nerfacc 加速结构"""
        aabb = torch.tensor([
            -self.radius, -self.radius, -self.radius,
            self.radius, self.radius, self.radius
        ], dtype=torch.float32)
        
        # 前景 OccupancyGrid
        self.register_buffer('scene_aabb', aabb)
        
        if hasattr(cfg_model, 'nerfacc') and cfg_model.nerfacc.grid_prune:
            self.estimator = OccGridEstimator(
                roi_aabb=aabb,
                resolution=cfg_model.nerfacc.occ_grid.resolution,
            )
            
            # 背景 OccupancyGrid
            if cfg_model.background.enabled:
                self.estimator_bg = OccGridEstimator(
                    roi_aabb=aabb,
                    resolution=cfg_model.nerfacc.occ_grid.resolution_bg,
                )
            else:
                self.estimator_bg = None
        else:
            self.estimator = None
            self.estimator_bg = None
            
    def forward(self, data):
        """训练前向传播
        
        Args:
            data: 包含 rays_o, rays_d, rgb 等的字典
            
        Returns:
            output: 包含渲染结果的字典
        """
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        sample_idx = data.get('idx', None)
        
        output = self.render_rays(rays_o, rays_d, sample_idx=sample_idx, stratified=self.training)
        return output
    
    @torch.no_grad()
    def inference(self, data):
        """推理前向传播"""
        self.eval()
        rays_o = data['rays_o']
        rays_d = data['rays_d']
        sample_idx = data.get('idx', None)
        
        # 分块渲染避免 OOM
        chunk_size = 4096
        outputs = defaultdict(list)
        
        for i in range(0, rays_o.shape[0], chunk_size):
            rays_o_chunk = rays_o[i:i+chunk_size]
            rays_d_chunk = rays_d[i:i+chunk_size]
            
            output = self.render_rays(rays_o_chunk, rays_d_chunk, sample_idx=sample_idx, stratified=False)
            
            for k, v in output.items():
                if v is not None:
                    outputs[k].append(v.detach())
                    
        # 合并结果
        for k, v in outputs.items():
            outputs[k] = torch.cat(v, dim=0)
            
        return dict(outputs)
    
    def render_rays(self, rays_o, rays_d, sample_idx=None, stratified=True):
        """渲染光线
        
        Args:
            rays_o: [N, 3] 光线起点
            rays_d: [N, 3] 光线方向
            sample_idx: [N] 图像索引（用于外观嵌入）
            stratified: 是否使用分层采样
            
        Returns:
            output: 渲染结果字典
        """
        n_rays = rays_o.shape[0]
        rays_d = F.normalize(rays_d, dim=-1)
        
        # 获取外观嵌入
        app, app_outside = self._get_appearance_embedding(sample_idx, n_rays)
        
        if NERFACC_AVAILABLE and self.estimator is not None:
            output = self._render_rays_nerfacc(rays_o, rays_d, app, app_outside, stratified)
        else:
            output = self._render_rays_vanilla(rays_o, rays_d, app, app_outside, stratified)
            
        return output
    
    def _render_rays_nerfacc(self, rays_o, rays_d, app, app_outside, stratified):
        """使用 nerfacc 加速的光线渲染"""
        n_rays = rays_o.shape[0]
        device = rays_o.device
        
        # 定义 sigma_fn 用于 occupancy grid 更新
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sdf = self.neural_sdf.sdf(positions)
            
            inv_s = self.s_var.exp().clamp(1e-6, 1e6)
            estimated_next_sdf = sdf - self.render_step_size * 0.5
            estimated_prev_sdf = sdf + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)
            return alpha
        
        # 使用 estimator 进行采样
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o, rays_d,
            sigma_fn=sigma_fn,
            near_plane=0.0,
            far_plane=1e10,
            render_step_size=self.render_step_size,
            stratified=stratified,
            alpha_thre=0.0,
        )
        
        # 前景渲染
        output = self._render_samples_foreground(
            rays_o, rays_d, ray_indices, t_starts, t_ends, app, n_rays
        )
        
        # 背景渲染
        if self.background_nerf is not None and self.estimator_bg is not None:
            output_bg = self._render_background_nerfacc(rays_o, rays_d, app_outside, stratified)
            output = self._combine_fg_bg(output, output_bg, n_rays)
            
        return output
    
    def _render_samples_foreground(self, rays_o, rays_d, ray_indices, t_starts, t_ends, app, n_rays):
        """渲染前景采样点"""
        if len(ray_indices) == 0:
            # 无有效采样点
            device = rays_o.device
            return {
                'rgb': torch.zeros(n_rays, 3, device=device),
                'depth': torch.zeros(n_rays, 1, device=device),
                'opacity': torch.zeros(n_rays, 1, device=device),
                'normal': torch.zeros(n_rays, 3, device=device),
                'num_samples': torch.tensor([0], device=device),
            }
            
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * midpoints[:, None]
        dists = t_ends - t_starts
        
        # SDF 和特征
        sdf, feat = self.neural_sdf(positions, with_sdf=True, with_feat=True)
        
        # 梯度和法向量
        gradients, hessians = self.neural_sdf.compute_gradients(positions, training=self.training, sdf=sdf)
        normals = F.normalize(gradients, dim=-1)
        
        # RGB
        if app is not None:
            app_samples = app[ray_indices]
        else:
            app_samples = None
        rgb = self.neural_rgb(positions, normals, t_dirs, feat, app=app_samples)
        
        # NeuS alpha 计算
        alpha = self._compute_neus_alpha(sdf, normals, t_dirs, dists)
        
        # 体积渲染
        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, values=None, ray_indices=ray_indices, n_rays=n_rays)
        depth = accumulate_along_rays(weights, values=midpoints[:, None], ray_indices=ray_indices, n_rays=n_rays)
        comp_rgb = accumulate_along_rays(weights, values=rgb, ray_indices=ray_indices, n_rays=n_rays)
        comp_normal = accumulate_along_rays(weights, values=normals, ray_indices=ray_indices, n_rays=n_rays)
        
        output = {
            'rgb': comp_rgb,
            'depth': depth,
            'opacity': opacity,
            'normal': F.normalize(comp_normal, dim=-1),
            'num_samples': torch.tensor([len(t_starts)], device=rays_o.device),
            # 训练用
            'sdf_samples': sdf,
            'gradients': gradients,
            'hessians': hessians,
            'weights': weights,
            'ray_indices': ray_indices,
        }
        
        return output
    
    def _render_rays_vanilla(self, rays_o, rays_d, app, app_outside, stratified):
        """Vanilla 光线渲染（不使用 nerfacc）"""
        n_rays = rays_o.shape[0]
        device = rays_o.device
        
        # 计算近远平面
        near, far = self._get_near_far(rays_o, rays_d)
        
        # 均匀采样
        num_samples = self.cfg.render.num_samples_per_ray
        t_vals = torch.linspace(0, 1, num_samples, device=device)
        t_vals = near[:, None] + (far - near)[:, None] * t_vals[None, :]
        
        if stratified and self.training:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
            lower = torch.cat([t_vals[..., :1], mids], dim=-1)
            t_rand = torch.rand_like(t_vals)
            t_vals = lower + (upper - lower) * t_rand
            
        # 采样点
        positions = rays_o[:, None, :] + rays_d[:, None, :] * t_vals[..., None]
        positions = positions.view(-1, 3)
        
        # SDF 和特征
        sdf, feat = self.neural_sdf(positions, with_sdf=True, with_feat=True)
        sdf = sdf.view(n_rays, num_samples, 1)
        feat = feat.view(n_rays, num_samples, -1)
        
        # 梯度和法向量
        gradients, hessians = self.neural_sdf.compute_gradients(positions, training=self.training, sdf=sdf.view(-1, 1))
        gradients = gradients.view(n_rays, num_samples, 3)
        if hessians is not None:
            hessians = hessians.view(n_rays, num_samples, 3)
        normals = F.normalize(gradients, dim=-1)
        
        # RGB
        rays_d_exp = rays_d[:, None, :].expand(-1, num_samples, -1).reshape(-1, 3)
        if app is not None:
            app_exp = app[:, None, :].expand(-1, num_samples, -1).reshape(-1, app.shape[-1])
        else:
            app_exp = None
        rgb = self.neural_rgb(positions, normals.view(-1, 3), rays_d_exp, feat.view(-1, feat.shape[-1]), app=app_exp)
        rgb = rgb.view(n_rays, num_samples, 3)
        
        # NeuS alpha 计算
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([dists, torch.full_like(dists[..., :1], 1e10)], dim=-1)
        alpha = self._compute_neus_alpha_batch(sdf[..., 0], normals, rays_d[:, None, :].expand(-1, num_samples, -1), dists)
        
        # 体积渲染
        weights = self._alpha_to_weights(alpha)
        
        comp_rgb = (weights[..., None] * rgb).sum(dim=1)
        depth = (weights * t_vals).sum(dim=1, keepdim=True)
        opacity = weights.sum(dim=1, keepdim=True)
        comp_normal = (weights[..., None] * normals).sum(dim=1)
        comp_normal = F.normalize(comp_normal, dim=-1)
        
        output = {
            'rgb': comp_rgb,
            'depth': depth,
            'opacity': opacity,
            'normal': comp_normal,
            'num_samples': torch.tensor([num_samples * n_rays], device=device),
            'sdf_samples': sdf,
            'gradients': gradients,
            'hessians': hessians,
        }
        
        # 背景
        if self.background_nerf is not None:
            output_bg = self._render_background_vanilla(rays_o, rays_d, far, app_outside, stratified)
            output = self._combine_fg_bg_vanilla(output, output_bg)
            
        return output
    
    def _compute_neus_alpha(self, sdf, normals, dirs, dists):
        """计算 NeuS 风格的 alpha 值"""
        inv_s = self.s_var.exp().clamp(1e-6, 1e6)
        
        true_cos = (dirs * normals).sum(dim=-1, keepdim=True)
        iter_cos = self._get_iter_cos(true_cos)
        
        estimated_next_sdf = sdf + iter_cos * dists[:, None] * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists[:, None] * 0.5
        
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        
        alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)
        return alpha.squeeze(-1)
    
    def _compute_neus_alpha_batch(self, sdf, normals, dirs, dists):
        """批量计算 NeuS alpha"""
        inv_s = self.s_var.exp().clamp(1e-6, 1e6)
        
        true_cos = (dirs * normals).sum(dim=-1)
        iter_cos = self._get_iter_cos(true_cos)
        
        estimated_next_sdf = sdf + iter_cos * dists * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists * 0.5
        
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        
        alpha = ((prev_cdf - next_cdf) / (prev_cdf + 1e-5)).clamp(0.0, 1.0)
        return alpha
    
    def _get_iter_cos(self, true_cos):
        """NeuS cosine 退火策略"""
        anneal_ratio = min(self.progress / self.anneal_end, 1.0)
        return -((-true_cos * 0.5 + 0.5).relu() * (1.0 - anneal_ratio) +
                 (-true_cos).relu() * anneal_ratio)
    
    def _alpha_to_weights(self, alpha):
        """将 alpha 转换为渲染权重"""
        T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
        T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)
        weights = alpha * T
        return weights
    
    def _get_near_far(self, rays_o, rays_d):
        """计算与场景边界的交点"""
        # 简化：使用固定的近远平面
        near = torch.zeros(rays_o.shape[0], device=rays_o.device)
        far = torch.ones(rays_o.shape[0], device=rays_o.device) * self.radius * 2
        return near, far
    
    def _get_appearance_embedding(self, sample_idx, n_rays):
        """获取外观嵌入"""
        if self.appear_embed is None or sample_idx is None:
            return None, None
            
        app = self.appear_embed(sample_idx)
        if self.appear_embed_outside is not None:
            app_outside = self.appear_embed_outside(sample_idx)
        else:
            app_outside = None
            
        return app, app_outside
    
    def _render_background_nerfacc(self, rays_o, rays_d, app_outside, stratified):
        """使用 nerfacc 渲染背景"""
        # 简化实现：直接返回白色或黑色背景
        n_rays = rays_o.shape[0]
        device = rays_o.device
        
        if self.white_background:
            bg_color = torch.ones(n_rays, 3, device=device)
        else:
            bg_color = torch.zeros(n_rays, 3, device=device)
            
        return {
            'rgb': bg_color,
            'opacity': torch.zeros(n_rays, 1, device=device),
        }
    
    def _render_background_vanilla(self, rays_o, rays_d, far, app_outside, stratified):
        """Vanilla 背景渲染"""
        n_rays = rays_o.shape[0]
        device = rays_o.device
        
        if self.white_background:
            bg_color = torch.ones(n_rays, 3, device=device)
        else:
            bg_color = torch.zeros(n_rays, 3, device=device)
            
        return {
            'rgb': bg_color,
            'opacity': torch.zeros(n_rays, 1, device=device),
        }
    
    def _combine_fg_bg(self, output_fg, output_bg, n_rays):
        """合并前景和背景"""
        fg_opacity = output_fg['opacity']
        comp_rgb = output_fg['rgb'] + output_bg['rgb'] * (1 - fg_opacity)
        
        output_fg['rgb'] = comp_rgb
        output_fg['rgb_fg'] = output_fg['rgb']
        output_fg['rgb_bg'] = output_bg['rgb']
        
        return output_fg
    
    def _combine_fg_bg_vanilla(self, output_fg, output_bg):
        """Vanilla 前景背景合并"""
        return self._combine_fg_bg(output_fg, output_bg, output_fg['rgb'].shape[0])
    
    def update_step(self, epoch, global_step):
        """更新训练步骤"""
        self.neural_sdf.update_step(epoch, global_step)
        
        # 更新 occupancy grid
        if self.estimator is not None:
            def occ_eval_fn(x):
                sdf = self.neural_sdf.sdf(x)
                inv_s = self.s_var.exp().clamp(1e-6, 1e6)
                alpha = (1 - torch.sigmoid(sdf * inv_s)).clamp(0, 1)
                return alpha
                
            self.estimator.update_every_n_steps(
                step=global_step,
                occ_eval_fn=occ_eval_fn,
                occ_thre=0.01,
            )
    
    @torch.no_grad()
    def export_mesh(self, resolution=512, threshold=0.0):
        """导出网格"""
        import mcubes
        import numpy as np
        
        # 创建网格
        x = torch.linspace(-self.radius, self.radius, resolution)
        y = torch.linspace(-self.radius, self.radius, resolution)
        z = torch.linspace(-self.radius, self.radius, resolution)
        
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).to(next(self.parameters()).device)
        
        # 分块计算 SDF
        chunk_size = 65536
        sdf_values = []
        for i in range(0, points.shape[0], chunk_size):
            sdf_chunk = self.neural_sdf.sdf(points[i:i+chunk_size])
            sdf_values.append(sdf_chunk.cpu())
            
        sdf = torch.cat(sdf_values, dim=0).reshape(resolution, resolution, resolution).numpy()
        
        # Marching cubes
        vertices, triangles = mcubes.marching_cubes(sdf, threshold)
        
        # 缩放回原始坐标
        vertices = vertices / (resolution - 1) * 2 * self.radius - self.radius
        
        return vertices, triangles
