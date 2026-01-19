"""Checkpointer for model saving and loading"""

import os
import torch
import threading


def to_cpu(state_dict):
    """将状态字典移动到 CPU"""
    return {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}


class Checkpointer:
    """检查点管理器
    
    用于保存和加载模型状态、优化器状态和调度器状态
    """
    
    def __init__(self, model, optim=None, sched=None):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.resume = False
        self.resume_epoch = None
        self.resume_iteration = None
        
    def save(self, checkpoint_path, current_epoch, current_iteration):
        """保存检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            current_epoch: 当前 epoch
            current_iteration: 当前迭代次数
            
        Returns:
            checkpoint_path: 保存的路径
        """
        save_dict = to_cpu(self._collect_state_dicts())
        save_dict.update(
            epoch=current_epoch,
            iteration=current_iteration,
        )
        
        # 异步保存
        threading.Thread(
            target=self._save_worker,
            daemon=False,
            args=(save_dict, checkpoint_path)
        ).start()
        
        return checkpoint_path
    
    def _save_worker(self, save_dict, checkpoint_path):
        """在后台线程中保存检查点"""
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(save_dict, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
        
    def _collect_state_dicts(self):
        """收集所有状态字典"""
        return dict(
            model=self.model.state_dict(),
            optim=self.optim.state_dict() if self.optim is not None else None,
            sched=self.sched.state_dict() if self.sched is not None else None,
        )
    
    def load(self, checkpoint_path, load_opt=True, load_sch=True, 
             iteration_mode=True, strict_resume=True):
        """加载检查点
        
        Args:
            checkpoint_path: 检查点文件路径
            load_opt: 是否加载优化器状态
            load_sch: 是否加载调度器状态
            iteration_mode: 是否使用迭代模式
            strict_resume: 是否严格加载权重
        """
        if checkpoint_path is None:
            print('No checkpoint path provided. Training from scratch.')
            torch.cuda.empty_cache()
            return
            
        if not os.path.exists(checkpoint_path):
            print(f'Warning: Checkpoint file not found: {checkpoint_path}')
            print('Training from scratch...')
            torch.cuda.empty_cache()
            return
            
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            
            # 加载模型
            print('- Loading model...')
            self.model.load_state_dict(state_dict['model'], strict=strict_resume)
            
            # 恢复状态
            self.resume = True
            self.resume_epoch = state_dict['epoch']
            self.resume_iteration = state_dict['iteration']
            
            if self.sched is not None:
                self.sched.last_epoch = self.resume_iteration if iteration_mode else self.resume_epoch
                
            if load_opt and self.optim is not None and state_dict.get('optim') is not None:
                print('- Loading optimizer...')
                self.optim.load_state_dict(state_dict['optim'])
                
            if load_sch and self.sched is not None and state_dict.get('sched') is not None:
                print('- Loading scheduler...')
                self.sched.load_state_dict(state_dict['sched'])
                
            print(f"Done loading checkpoint (epoch {self.resume_epoch}, iter {self.resume_iteration})")
            
        except Exception as e:
            print(f'Warning: Failed to load checkpoint: {checkpoint_path}')
            print(f'Error: {e}')
            print('Training from scratch...')
            self.resume = False
            self.resume_epoch = None
            self.resume_iteration = None
            
        torch.cuda.empty_cache()
