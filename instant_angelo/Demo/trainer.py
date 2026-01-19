"""Demo script for training Instant-Angelo"""

import os
import argparse
import torch

from instant_angelo.Config import Config
from instant_angelo.Module import Trainer
from instant_angelo.Util.misc import seed_everything


def main():
    parser = argparse.ArgumentParser(description='Train Instant-Angelo')
    parser.add_argument('--data_root', type=str, required=True, help='Path to data directory')
    parser.add_argument('--logdir', type=str, default='./exp', help='Output directory')
    parser.add_argument('--max_steps', type=int, default=20000, help='Maximum training steps')
    parser.add_argument('--resume', type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 设置随机种子
    seed_everything(args.seed)
    
    # 创建配置
    cfg = Config()
    cfg.data.root = args.data_root
    cfg.logdir = args.logdir
    cfg.trainer.max_steps = args.max_steps
    cfg.seed = args.seed
    
    # 自动设置 iters_per_epoch
    cfg.max_epoch = 20
    cfg.trainer.iters_per_epoch = args.max_steps // cfg.max_epoch
    
    # 创建输出目录
    os.makedirs(cfg.logdir, exist_ok=True)
    
    # 创建训练器
    trainer = Trainer(cfg, device=device)
    
    # 加载检查点
    if args.resume:
        trainer.load_checkpoint(args.resume)
        
    # 开始训练
    trainer.train()
    
    # 导出网格
    mesh_path = os.path.join(cfg.logdir, 'mesh.obj')
    trainer.export_mesh(mesh_path, resolution=512)
    
    print('Done!')


if __name__ == '__main__':
    main()
