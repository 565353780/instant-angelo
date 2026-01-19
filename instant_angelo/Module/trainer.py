import os
import torch
import inspect
import torch.nn.functional as F

from tqdm import tqdm
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from instant_angelo.Model import NeuSModel
from instant_angelo.Loss import eikonal_loss, curvature_loss, render_loss
from instant_angelo.Dataset.dataloader import get_train_dataloader, get_val_dataloader, cycle_dataloader
from .checkpointer import Checkpointer


def _calculate_model_size(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def requires_grad(model, flag=True):
    """设置模型梯度需求"""
    for p in model.parameters():
        p.requires_grad = flag


class Trainer:
    """Instant-Angelo 训练器
    
    Features:
    - 基于 epoch 的训练循环
    - 支持 nerfacc 加速
    - 自动混合精度训练
    - TensorBoard 日志
    - 检查点保存/恢复
    
    Args:
        cfg: 配置对象
        device: 训练设备
    """
    
    def __init__(self, cfg, device='cuda'):
        print('Setting up Instant-Angelo Trainer...')
        
        self.cfg = cfg
        self.device = device
        
        # 初始化 cuDNN
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 创建模型、优化器、调度器
        self.model = self._setup_model()
        self.optim = self._setup_optimizer()
        self.sched = self._setup_scheduler()
        
        # 自动混合精度
        self._init_amp()
        
        # 损失函数
        self._init_losses()
        
        # 检查点管理
        self.checkpointer = Checkpointer(self.model, self.optim, self.sched)
        self.checkpoint_path_last = os.path.join(cfg.logdir, 'model_last.pt')
        self.checkpoint_path_best = os.path.join(cfg.logdir, 'model_best.pt')
        self.best_psnr = float('-inf')
        
        # 训练状态
        self.current_epoch = 0
        self.current_iteration = 0
        self.progress = 0.0
        
        # 每个 epoch 的迭代次数
        self.iters_per_epoch = cfg.iters_per_epoch
        
        # 损失和指标
        self.losses = {}
        self.metrics = {}
        
        # 数据加载器
        self.train_loader = get_train_dataloader(cfg)
        self.val_loader = get_val_dataloader(cfg)
        
        # TensorBoard
        self._init_tensorboard()
        
        print(f'Model parameter count: {_calculate_model_size(self.model):,}')
        
    def _setup_model(self):
        """创建模型"""
        model = NeuSModel(self.cfg.model, self.cfg.data)
        model = model.to(self.device)
        return model
    
    def _setup_optimizer(self):
        """创建优化器"""
        optim = AdamW(
            params=self.model.parameters(),
            lr=self.cfg.optim.params.lr,
            weight_decay=self.cfg.optim.params.weight_decay,
            betas=tuple(self.cfg.optim.params.betas),
            eps=self.cfg.optim.params.eps,
        )
        
        self.optim_zero_grad_kwargs = {}
        if 'set_to_none' in inspect.signature(optim.zero_grad).parameters:
            self.optim_zero_grad_kwargs['set_to_none'] = True
            
        return optim
    
    def _setup_scheduler(self):
        """创建学习率调度器"""
        warm_up_end = self.cfg.optim.sched.warm_up_end
        two_steps = self.cfg.optim.sched.two_steps
        gamma = self.cfg.optim.sched.gamma
        
        def lr_lambda(x):
            if x < warm_up_end:
                return x / warm_up_end
            elif x > two_steps[1]:
                return 1.0 / gamma ** 2
            elif x > two_steps[0]:
                return 1.0 / gamma
            else:
                return 1.0
                
        return LambdaLR(self.optim, lr_lambda)
    
    def _init_amp(self):
        """初始化自动混合精度"""
        if getattr(self.cfg.trainer, 'allow_tf32', True):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
        scaler_kwargs = {
            'enabled': self.cfg.trainer.amp_config.enabled,
            'init_scale': self.cfg.trainer.scaler_config.init_scale,
            'growth_factor': self.cfg.trainer.scaler_config.growth_factor,
            'backoff_factor': self.cfg.trainer.scaler_config.backoff_factor,
            'growth_interval': self.cfg.trainer.scaler_config.growth_interval,
        }
        
        self.scaler = GradScaler('cuda', **scaler_kwargs)
        
    def _init_losses(self):
        """初始化损失函数"""
        self.weights = {}
        loss_weight = self.cfg.trainer.loss_weight
        
        for attr in ['render', 'eikonal', 'curvature', 'sparsity', 'distortion', 'mask', 'sdf_prior']:
            weight = getattr(loss_weight, attr, 0.0)
            if weight > 0:
                self.weights[attr] = weight
                print(f'Loss {attr:<20} Weight {weight}')

    def _init_tensorboard(self):
        """初始化 TensorBoard"""
        from datetime import datetime
        tb_dir = os.path.join(self.cfg.logdir, datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)

    def load_checkpoint(self, checkpoint_path=None, load_opt=True, load_sch=True):
        """加载检查点"""
        if checkpoint_path is None and os.path.exists(self.checkpoint_path_last):
            checkpoint_path = self.checkpoint_path_last
            
        self.checkpointer.load(
            checkpoint_path,
            load_opt=load_opt,
            load_sch=load_sch,
            iteration_mode=self.cfg.optim.sched.iteration_mode,
            strict_resume=self.cfg.checkpoint.strict_resume,
        )
        
    def train(self):
        """主训练循环"""
        # 恢复训练状态
        start_epoch = self.checkpointer.resume_epoch or 0
        self.current_iteration = self.checkpointer.resume_iteration or 0

        # 初始验证
        print('Initial validation...')
        self._validate()

        # 创建数据迭代器
        data_iter = cycle_dataloader(self.train_loader)

        # 主训练循环
        for epoch in range(start_epoch, self.cfg.max_epoch):
            self.current_epoch = epoch
            self._start_of_epoch(epoch)
            
            # Epoch 进度条
            pbar = tqdm(
                range(self.iters_per_epoch),
                desc=f'Epoch {epoch + 1}/{self.cfg.max_epoch}',
                leave=False
            )
            
            for it in pbar:
                # 获取数据
                data = next(data_iter)
                data = self._prepare_data(data)
                
                # 训练步骤
                last_iter = (it == self.iters_per_epoch - 1)
                self._train_step(data, last_iter)
                
                self.current_iteration += 1
                pbar.set_postfix(
                    iter=self.current_iteration,
                    loss=f'{self.losses.get("total", 0):.4f}'
                )
                
                # 更新调度器
                if self.cfg.optim.sched.iteration_mode:
                    self.sched.step()
                    
            # Epoch 结束
            self._end_of_epoch(epoch)
            
        # 训练完成
        self.checkpointer.save(self.checkpoint_path_last, self.cfg.max_epoch, self.current_iteration)
        self.writer.close()
        print('Training completed!')
        
    def _start_of_epoch(self, epoch):
        """Epoch 开始时的操作"""
        self.model.train()
        
    def _prepare_data(self, data):
        """准备训练数据"""
        data = to_cuda(data, self.device)
        
        # 从数据中采样光线
        batch_size = data['image'].shape[0]
        num_rays = self.cfg.model.render.rand_rays
        
        # 假设 batch_size = 1
        dataset = self.train_loader.dataset
        sample = dataset.sample_rays(data['idx'][0].item(), num_rays)
        
        return to_cuda(sample, self.device)
    
    def _train_step(self, data, last_iter_in_epoch=False):
        """单步训练"""
        requires_grad(self.model, True)
        
        # 更新进度
        max_iter = self.cfg.max_epoch * self.iters_per_epoch
        self.progress = self.current_iteration / max_iter
        self.model.progress = self.progress
        
        # 更新模型
        self.model.update_step(self.current_epoch, self.current_iteration)
        
        # 自动混合精度
        amp_dtype = torch.float16 if self.cfg.trainer.amp_config.dtype == 'float16' else torch.bfloat16
        
        with autocast('cuda', enabled=self.cfg.trainer.amp_config.enabled, dtype=amp_dtype):
            # 前向传播
            output = self.model({
                'rays_o': data['rays_o'],
                'rays_d': data['rays_d'],
                'idx': data['idx'] if isinstance(data['idx'], torch.Tensor) else torch.tensor([data['idx']], device=self.device),
            })
            
            # 计算损失
            total_loss = self._compute_loss(data, output)
            total_loss = total_loss / float(self.cfg.trainer.grad_accum_iter)
            
        # 反向传播
        self.scaler.scale(total_loss).backward()
        
        # 优化器步骤
        if (self.current_iteration + 1) % self.cfg.trainer.grad_accum_iter == 0 or last_iter_in_epoch:
            self.scaler.step(self.optim)
            self.scaler.update()
            self.optim.zero_grad(**self.optim_zero_grad_kwargs)
            
        # 分离损失
        for name in self.losses:
            self.losses[name] = self.losses[name].detach()
            
    def _compute_loss(self, data, output):
        """计算损失"""
        total_loss = torch.tensor(0., device=self.device)
        
        # 渲染损失
        if 'render' in self.weights:
            loss_render = render_loss(output['rgb'], data['rgb'])
            self.losses['render'] = loss_render
            total_loss += loss_render * self.weights['render']
            
        # PSNR
        with torch.no_grad():
            mse = F.mse_loss(output['rgb'], data['rgb'])
            self.metrics['psnr'] = -10 * torch.log10(mse)
            
        # Eikonal 损失
        if 'eikonal' in self.weights and 'gradients' in output:
            loss_eikonal = eikonal_loss(output['gradients'])
            self.losses['eikonal'] = loss_eikonal
            total_loss += loss_eikonal * self.weights['eikonal']
            
        # 曲率损失
        if 'curvature' in self.weights and output.get('hessians') is not None:
            weight = self._get_curvature_weight()
            loss_curvature = curvature_loss(output['hessians'])
            self.losses['curvature'] = loss_curvature
            total_loss += loss_curvature * weight
            
        self.losses['total'] = total_loss
        return total_loss
    
    def _get_curvature_weight(self):
        """获取曲率损失权重（随训练进度调整）"""
        init_weight = self.weights.get('curvature', 0.0)
        warm_up_end = self.cfg.optim.sched.warm_up_end
        
        if self.current_iteration <= warm_up_end:
            return self.current_iteration / warm_up_end * init_weight
        else:
            # 随 coarse-to-fine 级别衰减
            decay = self.model.neural_sdf.encoding.per_level_scale ** (self.model.neural_sdf.active_levels - 1)
            return init_weight / decay
            
    def _end_of_epoch(self, epoch):
        """Epoch 结束时的操作"""
        # 更新调度器
        if not self.cfg.optim.sched.iteration_mode:
            self.sched.step()
            
        # 日志
        print(f'Epoch {epoch + 1}, Iter {self.current_iteration}, Loss: {self.losses.get("total", 0):.4f}')
        self._log_tensorboard('train')
        
        # 验证
        self._validate()
        
        # 保存最佳模型
        current_psnr = self.metrics.get('psnr', torch.tensor(0.)).item()
        if current_psnr > self.best_psnr:
            self.best_psnr = current_psnr
            self.checkpointer.save(self.checkpoint_path_best, epoch + 1, self.current_iteration)
            print(f'New best model! PSNR: {current_psnr:.4f}')
            
        # 保存最新模型
        self.checkpointer.save(self.checkpoint_path_last, epoch + 1, self.current_iteration)
        
    @torch.no_grad()
    def _validate(self):
        """验证"""
        self.model.eval()
        
        total_psnr = 0
        count = 0
        
        for i, data in enumerate(tqdm(self.val_loader, desc='Validating', leave=False)):
            if i >= self.cfg.data.val.max_viz_samples:
                break
                
            data = to_cuda(data, self.device)
            
            # 推理
            output = self.model.inference({
                'rays_o': data['rays_o'][0],
                'rays_d': data['rays_d'][0],
                'idx': data['idx'],
            })
            
            # 计算 PSNR
            rgb_pred = output['rgb'].view(self.val_loader.dataset.dataset.H, 
                                         self.val_loader.dataset.dataset.W, 3)
            rgb_gt = data['image'][0].permute(1, 2, 0)
            
            mse = F.mse_loss(rgb_pred, rgb_gt)
            psnr = -10 * torch.log10(mse)
            total_psnr += psnr.item()
            count += 1
            
            # 保存可视化
            if i == 0:
                self._log_images(rgb_pred, rgb_gt, output)
                
        self.metrics['val_psnr'] = torch.tensor(total_psnr / max(count, 1))
        self._log_tensorboard('val')
        
        self.model.train()
        
    def _log_tensorboard(self, mode):
        """记录 TensorBoard"""
        # 损失
        for name, value in self.losses.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{mode}/loss/{name}', value, self.current_iteration)
            
        # 指标
        for name, value in self.metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f'{mode}/{name}', value, self.current_iteration)
            
        # 学习率
        if mode == 'train':
            self.writer.add_scalar('optim/lr', self.sched.get_last_lr()[0], self.current_iteration)
            self.writer.add_scalar('train/s_var', self.model.s_var.exp().item(), self.current_iteration)
            self.writer.add_scalar('train/active_levels', self.model.neural_sdf.active_levels, self.current_iteration)
            
    def _log_images(self, rgb_pred, rgb_gt, output):
        """记录图像到 TensorBoard"""
        # RGB
        rgb_pred_img = rgb_pred.permute(2, 0, 1).clamp(0, 1)
        rgb_gt_img = rgb_gt.permute(2, 0, 1).clamp(0, 1)
        error_img = (rgb_pred - rgb_gt).abs().permute(2, 0, 1)
        
        self.writer.add_image('val/rgb_pred', rgb_pred_img, self.current_iteration)
        self.writer.add_image('val/rgb_gt', rgb_gt_img, self.current_iteration)
        self.writer.add_image('val/error', error_img, self.current_iteration)
        
        # 深度
        if 'depth' in output:
            depth = output['depth'].view(rgb_pred.shape[0], rgb_pred.shape[1])
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            self.writer.add_image('val/depth', depth_normalized.unsqueeze(0), self.current_iteration)
            
        # 法向量
        if 'normal' in output:
            normal = output['normal'].view(rgb_pred.shape[0], rgb_pred.shape[1], 3)
            normal_img = (normal * 0.5 + 0.5).permute(2, 0, 1).clamp(0, 1)
            self.writer.add_image('val/normal', normal_img, self.current_iteration)
            
    @torch.no_grad()
    def export_mesh(self, save_path, resolution=512):
        """导出网格"""
        print(f'Exporting mesh to {save_path}...')
        self.model.eval()
        
        vertices, triangles = self.model.export_mesh(resolution=resolution)
        
        # 保存为 OBJ
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        mesh.export(save_path)
        
        print(f'Mesh exported: {len(vertices)} vertices, {len(triangles)} faces')
        return mesh
