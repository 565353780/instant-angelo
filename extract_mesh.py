#!/usr/bin/env python
"""Extract mesh from trained Instant-Angelo model"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instant_angelo import Config, NeuSModel
from instant_angelo.Module import Checkpointer


def main():
    parser = argparse.ArgumentParser(description='Extract mesh from Instant-Angelo')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output', type=str, default='mesh.obj', help='Output mesh path')
    parser.add_argument('--resolution', type=int, default=512, help='Mesh resolution')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    args = parser.parse_args()
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 加载检查点
    print(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location='cpu')
    
    # 创建模型
    cfg = Config()
    model = NeuSModel(cfg.model)
    model.load_state_dict(state_dict['model'])
    model = model.to(device)
    model.eval()
    
    # 导出网格
    print(f'Extracting mesh at resolution {args.resolution}...')
    vertices, triangles = model.export_mesh(resolution=args.resolution)
    
    # 保存
    import trimesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    mesh.export(args.output)
    
    print(f'Mesh saved to {args.output}')
    print(f'Vertices: {len(vertices)}, Faces: {len(triangles)}')


if __name__ == '__main__':
    main()
