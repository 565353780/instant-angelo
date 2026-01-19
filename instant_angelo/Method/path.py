"""Path utilities"""

import os


def ensure_dir(path):
    """确保目录存在"""
    os.makedirs(path, exist_ok=True)
    return path


def get_filename(path):
    """获取文件名（不含扩展名）"""
    return os.path.splitext(os.path.basename(path))[0]


def get_extension(path):
    """获取文件扩展名"""
    return os.path.splitext(path)[1]
