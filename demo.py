import os
import sys
import argparse

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instant_angelo import Config, Trainer
from instant_angelo.Util.misc import seed_everything


def demo_train():
    """训练示例"""
    parser = argparse.ArgumentParser(description='Instant-Angelo Demo')
    parser.add_argument('--data_root', type=str, required=True, help='Path to data directory')
    parser.add_argument('--logdir', type=str, default='./demo_output', help='Output directory')
    parser.add_argument('--quick', action='store_true', help='Quick demo (fewer iterations)')
    args = parser.parse_args()

    # 设置
    device = 'cuda:5'
    seed_everything(42)

    # 创建配置
    cfg = Config()
    cfg.data.root = args.data_root
    cfg.logdir = args.logdir

    if args.quick:
        # 快速演示配置
        cfg.trainer.max_steps = 1000
        cfg.max_epoch = 2
        cfg.model.render.rand_rays = 128
        cfg.model.render.num_samples_per_ray = 256
    else:
        cfg.trainer.max_steps = 20000
        cfg.max_epoch = 20

    os.makedirs(cfg.logdir, exist_ok=True)

    # 创建训练器
    print('Creating trainer...')
    trainer = Trainer(cfg, device=device)

    # 训练
    print('Starting training...')
    trainer.train()

    # 导出网格
    print('Exporting mesh...')
    mesh_path = os.path.join(cfg.logdir, 'mesh.obj')
    trainer.export_mesh(mesh_path, resolution=256 if args.quick else 512)

    print(f'Done! Results saved to {cfg.logdir}')


if __name__ == '__main__':
    demo_train()
