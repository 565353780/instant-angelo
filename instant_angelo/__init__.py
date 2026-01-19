"""Instant-Angelo: Fast Neural Surface Reconstruction

基于 neural-angelo 的代码结构，支持 nerfacc 加速框架
"""

from .Config import Config, load_config_from_yaml
from .Model import NeuSModel, NeuralSDF, NeuralRGB
from .Module import Trainer, Checkpointer

__version__ = '0.1.0'
__all__ = [
    'Config',
    'load_config_from_yaml',
    'NeuSModel',
    'NeuralSDF',
    'NeuralRGB',
    'Trainer',
    'Checkpointer',
]
