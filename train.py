#!/usr/bin/env python
"""Train Instant-Angelo

使用方法:
    python train.py --data_root /path/to/data --logdir ./exp
    
或者使用 YAML 配置:
    python train.py --config configs/instant_angelo.yaml
"""

import os
import sys
import argparse
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instant_angelo.Config import Config, load_config_from_yaml
from instant_angelo.Module import Trainer
from instant_angelo.Util.misc import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Train Instant-Angelo')
    
    # 配置选项
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config file')
    parser.add_argument('--data_root', type=str, default=None, help='Path to data directory')
    parser.add_argument('--logdir', type=str, default='./exp', help='Output directory')
    
    # 训练选项
    parser.add_argument('--max_steps', type=int, default=20000, help='Maximum training steps')
    parser.add_argument('--max_epoch', type=int, default=20, help='Maximum epochs')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    
    # 模型选项
    parser.add_argument('--radius', type=float, default=1.5, help='Scene radius')
    parser.add_argument('--num_samples', type=int, default=512, help='Number of samples per ray')
    parser.add_argument('--rand_rays', type=int, default=256, help='Number of rays per iteration')
    
    # 系统选项
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    
    # 导出选项
    parser.add_argument('--export_mesh', action='store_true', help='Export mesh after training')
    parser.add_argument('--mesh_resolution', type=int, default=512, help='Mesh resolution')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 加载配置
    if args.config is not None and os.path.exists(args.config):
        print(f'Loading config from {args.config}')
        cfg = load_config_from_yaml(args.config)
    else:
        cfg = Config()
        
    # 覆盖命令行参数
    if args.data_root is not None:
        cfg.data.root = args.data_root
    cfg.logdir = args.logdir
    cfg.seed = args.seed
    
    if args.max_steps is not None:
        cfg.trainer.max_steps = args.max_steps
    if args.max_epoch is not None:
        cfg.max_epoch = args.max_epoch
        
    if args.radius is not None:
        cfg.model.radius = args.radius
    if args.num_samples is not None:
        cfg.model.render.num_samples_per_ray = args.num_samples
    if args.rand_rays is not None:
        cfg.model.render.rand_rays = args.rand_rays
        
    # 验证配置
    if cfg.data.root is None:
        print('Error: --data_root is required')
        sys.exit(1)
        
    if not os.path.exists(cfg.data.root):
        print(f'Error: Data directory not found: {cfg.data.root}')
        sys.exit(1)
        
    # 创建输出目录
    os.makedirs(cfg.logdir, exist_ok=True)
    
    print(f'Data root: {cfg.data.root}')
    print(f'Log directory: {cfg.logdir}')
    print(f'Max steps: {cfg.trainer.max_steps}')
    print(f'Max epochs: {cfg.max_epoch}')
    
    # 创建训练器
    trainer = Trainer(cfg, device=device)
    
    # 加载检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # 开始训练
    trainer.train()
    
    # 导出网格
    if args.export_mesh:
        mesh_path = os.path.join(cfg.logdir, 'mesh.obj')
        trainer.export_mesh(mesh_path, resolution=args.mesh_resolution)
        
    print('Training completed!')


if __name__ == '__main__':
    main()
