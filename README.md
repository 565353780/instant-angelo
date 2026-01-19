# Instant-Angelo

基于 nerfacc 加速框架的快速神经表面重建

## 特性

- **高效的 HashGrid 编码**: 使用 tiny-cuda-nn 的多分辨率哈希网格
- **nerfacc 加速**: 支持最新版 nerfacc 的 OccupancyGrid 剪枝
- **Coarse-to-Fine 训练**: 渐进式激活哈希网格级别
- **模块化设计**: 按照 neural-angelo 的代码结构组织

## 安装

```bash
conda create -n instant-angelo python=3.10
conda activate instant-angelo
./setup.sh
```

## 代码结构

```
instant_angelo/
├── Config/          # 配置类
│   └── config.py
├── Dataset/         # 数据加载
│   ├── data.py
│   └── dataloader.py
├── Model/           # 模型
│   ├── neural_sdf.py   # SDF 网络
│   ├── neural_rgb.py   # RGB 网络
│   ├── background_nerf.py  # 背景网络
│   └── model.py        # NeuS 主模型
├── Loss/            # 损失函数
│   ├── eikonal.py
│   ├── curvature.py
│   └── render.py
├── Module/          # 训练模块
│   ├── trainer.py      # 训练器
│   └── checkpointer.py # 检查点管理
├── Method/          # 工具方法
│   ├── io.py
│   ├── path.py
│   ├── time.py
│   └── data.py
└── Util/            # 工具函数
    ├── misc.py
    ├── visualization.py
    └── mesh.py
```

## 使用方法

### 训练

```bash
# 基本用法
python train.py --data_root /path/to/data --logdir ./exp

# 快速演示
python demo.py --data_root /path/to/data --quick

# 使用自定义配置
python train.py --config configs/instant_angelo.yaml
```

### 导出网格

```bash
python extract_mesh.py --checkpoint ./exp/model_best.pt --output mesh.obj --resolution 512
```

## 数据格式

支持 Instant-NGP/NeRF 风格的 `transforms.json` 格式:

```json
{
    "fl_x": 1000,
    "fl_y": 1000,
    "cx": 400,
    "cy": 300,
    "sphere_center": [0, 0, 0],
    "sphere_radius": 1.0,
    "frames": [
        {
            "file_path": "images/0001.png",
            "transform_matrix": [[...], [...], [...], [...]]
        }
    ]
}
```

## Trainer API

```python
from instant_angelo import Config, Trainer

# 创建配置
cfg = Config()
cfg.data.root = '/path/to/data'
cfg.logdir = './exp'
cfg.trainer.max_steps = 20000

# 创建训练器
trainer = Trainer(cfg, device='cuda')

# 训练
trainer.train()

# 导出网格
trainer.export_mesh('./mesh.obj', resolution=512)
```

## 扩展指南

### 自定义损失函数

在 `instant_angelo/Loss/` 中添加新的损失函数:

```python
# custom_loss.py
def my_custom_loss(pred, target):
    return torch.mean((pred - target) ** 2)
```

### 自定义模型

继承 `NeuSModel` 类:

```python
from instant_angelo.Model import NeuSModel

class MyModel(NeuSModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        # 添加自定义模块
```

## 依赖

- PyTorch >= 2.0
- nerfacc >= 0.5.0
- tiny-cuda-nn
- trimesh
- mcubes

## 致谢

- [Neuralangelo](https://github.com/NVlabs/neuralangelo)
- [nerfacc](https://github.com/nerfstudio-project/nerfacc)
- [instant-ngp](https://github.com/NVlabs/instant-ngp)

## License

MIT License
