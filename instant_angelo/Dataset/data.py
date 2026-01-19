"""Dataset for Instant-Angelo training"""

import os
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """COLMAP/NeRF 格式数据集
    
    支持 transforms.json 格式（类似 Instant-NGP）
    """
    
    def __init__(self, cfg, is_inference=False):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root
        self.is_inference = is_inference
        self.preload = getattr(cfg, 'preload', True)
        
        # 图像尺寸
        if is_inference:
            self.H, self.W = cfg.val.image_size
        else:
            self.H, self.W = cfg.train.image_size
            
        # 加载元数据
        meta_fname = os.path.join(self.root, 'transforms.json')
        with open(meta_fname, 'r') as f:
            self.meta = json.load(f)
            
        self.frames = self.meta['frames']
        
        # 子集采样
        if not is_inference and hasattr(cfg.train, 'subset') and cfg.train.subset:
            subset = cfg.train.subset
            subset_idx = np.linspace(0, len(self.frames), subset + 1)[:-1].astype(int)
            self.frames = [self.frames[i] for i in subset_idx]
            
        # 相机参数
        self.fx = self.meta.get('fl_x', self.meta.get('camera_angle_x', 0.5) * self.W / 2)
        self.fy = self.meta.get('fl_y', self.fx)
        self.cx = self.meta.get('cx', self.W / 2)
        self.cy = self.meta.get('cy', self.H / 2)
        
        # 场景归一化参数
        self.sphere_center = np.array(self.meta.get('sphere_center', [0, 0, 0]))
        self.sphere_radius = self.meta.get('sphere_radius', 1.0)
        
        # 调整参数
        if hasattr(cfg, 'readjust') and cfg.readjust is not None:
            self.sphere_center += np.array(getattr(cfg.readjust, 'center', [0, 0, 0]))
            self.sphere_radius *= getattr(cfg.readjust, 'scale', 1.0)
            
        # 预加载
        if self.preload:
            self._preload_data()
            
    def _preload_data(self):
        """预加载所有数据"""
        print(f"Preloading {len(self.frames)} images...")
        self.images = []
        self.poses = []
        
        for i, frame in enumerate(self.frames):
            # 加载图像
            img_path = os.path.join(self.root, frame['file_path'])
            if not img_path.endswith('.png') and not img_path.endswith('.jpg'):
                # 尝试添加扩展名
                for ext in ['.png', '.jpg', '.jpeg']:
                    if os.path.exists(img_path + ext):
                        img_path = img_path + ext
                        break
                        
            img = Image.open(img_path)
            img = img.resize((self.W, self.H), Image.BILINEAR)
            img = TF.to_tensor(img)[:3]  # RGB
            self.images.append(img)
            
            # 加载相机姿态
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            c2w = self._process_pose(c2w)
            self.poses.append(c2w)
            
        self.images = torch.stack(self.images, dim=0)
        self.poses = torch.stack(self.poses, dim=0)
        print(f"Preloaded {len(self.images)} images with shape {self.images.shape}")
        
    def _process_pose(self, c2w):
        """处理相机姿态
        
        - 转换 OpenGL -> OpenCV 坐标系
        - 应用场景归一化
        """
        # OpenGL -> OpenCV: 翻转 Y 和 Z 轴
        c2w[:3, 1:3] *= -1
        
        # 平移到场景中心
        c2w[:3, 3] -= torch.tensor(self.sphere_center, dtype=torch.float32)
        
        # 缩放
        c2w[:3, 3] /= self.sphere_radius
        
        return c2w[:3, :]  # [3, 4]
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        if self.preload:
            image = self.images[idx]
            pose = self.poses[idx]
        else:
            frame = self.frames[idx]
            
            # 加载图像
            img_path = os.path.join(self.root, frame['file_path'])
            if not os.path.exists(img_path):
                for ext in ['.png', '.jpg', '.jpeg']:
                    if os.path.exists(img_path + ext):
                        img_path = img_path + ext
                        break
                        
            image = Image.open(img_path)
            image = image.resize((self.W, self.H), Image.BILINEAR)
            image = TF.to_tensor(image)[:3]
            
            # 加载姿态
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)
            pose = self._process_pose(c2w)
            
        # 构建内参矩阵
        intr = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        sample = {
            'idx': idx,
            'image': image,  # [3, H, W]
            'pose': pose,    # [3, 4]
            'intr': intr,    # [3, 3]
        }
        
        return sample
    
    def get_rays(self, pose, intr):
        """生成光线
        
        Args:
            pose: [3, 4] 相机姿态 (camera-to-world)
            intr: [3, 3] 相机内参
            
        Returns:
            rays_o: [H*W, 3] 光线起点
            rays_d: [H*W, 3] 光线方向
        """
        device = pose.device
        
        # 像素坐标
        u = torch.arange(self.W, device=device)
        v = torch.arange(self.H, device=device)
        u, v = torch.meshgrid(u, v, indexing='xy')
        u = u.flatten()
        v = v.flatten()
        
        # 相机坐标系中的方向
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        
        dirs = torch.stack([
            (u - cx) / fx,
            (v - cy) / fy,
            torch.ones_like(u)
        ], dim=-1)  # [H*W, 3]
        
        # 转换到世界坐标系
        R = pose[:3, :3]  # [3, 3]
        t = pose[:3, 3]   # [3]
        
        rays_d = dirs @ R.T  # [H*W, 3]
        rays_o = t.expand(rays_d.shape[0], -1)  # [H*W, 3]
        
        return rays_o, rays_d
    
    def sample_rays(self, idx, num_rays):
        """随机采样光线
        
        Args:
            idx: 图像索引
            num_rays: 采样光线数
            
        Returns:
            sample: 包含采样光线的字典
        """
        sample = self[idx]
        
        # 获取所有光线
        rays_o, rays_d = self.get_rays(sample['pose'], sample['intr'])
        
        # 随机采样
        total_pixels = self.H * self.W
        ray_indices = torch.randperm(total_pixels)[:num_rays]
        
        # 采样像素颜色
        image_flat = sample['image'].permute(1, 2, 0).reshape(-1, 3)  # [H*W, 3]
        
        sample.update({
            'rays_o': rays_o[ray_indices],
            'rays_d': rays_d[ray_indices],
            'rgb': image_flat[ray_indices],
            'ray_indices': ray_indices,
        })
        
        return sample


class InferenceDataset(TorchDataset):
    """推理数据集，返回完整图像的光线"""
    
    def __init__(self, cfg):
        super().__init__()
        self.dataset = Dataset(cfg, is_inference=True)
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # 获取所有光线
        rays_o, rays_d = self.dataset.get_rays(sample['pose'], sample['intr'])
        
        sample.update({
            'rays_o': rays_o,
            'rays_d': rays_d,
        })
        
        return sample
