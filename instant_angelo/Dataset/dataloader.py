"""DataLoader utilities"""

import torch
from torch.utils.data import DataLoader


def get_train_dataloader(cfg, shuffle=True):
    """获取训练数据加载器"""
    from .data import Dataset
    
    dataset = Dataset(cfg.data, is_inference=False)
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    return loader


def get_val_dataloader(cfg):
    """获取验证数据加载器"""
    from .data import InferenceDataset
    
    dataset = InferenceDataset(cfg.data)
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )
    
    return loader


def cycle_dataloader(data_loader):
    """创建无限循环的数据迭代器"""
    while True:
        for data in data_loader:
            yield data
