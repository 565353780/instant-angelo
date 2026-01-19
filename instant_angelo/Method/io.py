"""I/O utilities"""

import os
import json
import yaml


def load_json(path):
    """加载 JSON 文件"""
    with open(path, 'r') as f:
        return json.load(f)


def save_json(data, path):
    """保存 JSON 文件"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_yaml(path):
    """加载 YAML 文件"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    """保存 YAML 文件"""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
