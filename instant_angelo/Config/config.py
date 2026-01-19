class Config:
    # ==================== 数据配置 ====================
    class Data:
        root: str = None
        preload: bool = True
        num_workers: int = 4
        num_images: int = None  # 自动从数据集获取

        class Train:
            image_size: list = [800, 800]  # [H, W]
            subset: int = None  # 如果设置，只使用部分数据

        train = Train()

        class Val:
            image_size: list = [300, 300]
            max_viz_samples: int = 4
            subset: int = 4

        val = Val()

        class Readjust:
            center: list = [0, 0, 0]
            scale: float = 1.0

        readjust = None  # 可选

    data = Data()

    # ==================== 训练器配置 ====================
    class Trainer:
        max_steps: int = 20000
        iters_per_epoch: int = 1000
        grad_accum_iter: int = 1
        depth_vis_scale: float = 1.0

        class Init:
            type: str = "none"  # "none", "xavier", "kaiming"
            gain: float = 1.0
            bias: float = None

        init = Init()

        class AmpConfig:
            enabled: bool = True
            dtype: str = "float16"  # "float16" or "bfloat16"

        amp_config = AmpConfig()

        class ScalerConfig:
            enabled: bool = True
            init_scale: float = 65536.0
            growth_factor: float = 2.0
            backoff_factor: float = 0.5
            growth_interval: int = 2000

        scaler_config = ScalerConfig()

        class EmaConfig:
            enabled: bool = False
            beta: float = 0.999
            start_iteration: int = 0

        ema_config = EmaConfig()

        class LossWeight:
            render: float = 1.0
            eikonal: float = 0.1
            curvature: float = 0.5
            sparsity: float = 0.0
            distortion: float = 0.0
            mask: float = 0.0
            sdf_prior: float = 0.0

        loss_weight = LossWeight()

        allow_tf32: bool = True

    trainer = Trainer()

    # ==================== 优化器配置 ====================
    class Optim:
        class Params:
            lr: float = 0.01
            weight_decay: float = 0.0
            betas: list = [0.9, 0.99]
            eps: float = 1e-15

        params = Params()

        class Sched:
            warm_up_end: int = 500
            two_steps: list = [15000, 18000]
            gamma: float = 10.0
            iteration_mode: bool = True

        sched = Sched()

    optim = Optim()

    # ==================== 模型配置 ====================
    class Model:
        radius: float = 1.5

        class Render:
            rand_rays: int = 256  # 每次迭代采样的光线数
            num_samples_per_ray: int = 512  # 前景采样点数
            num_samples_per_ray_bg: int = 256  # 背景采样点数
            stratified: bool = True

        render = Render()

        class Object:
            class SDF:
                class MLP:
                    num_layers: int = 2
                    hidden_dim: int = 64
                    skip: list = []

                    class ActivParams:
                        beta: float = 100.0

                    activ_params = ActivParams()
                    geometric_init: bool = True
                    weight_norm: bool = True
                    out_bias: float = 0.5
                    inside_out: bool = False

                mlp = MLP()

                class Encoding:
                    levels: int = 16

                    class Hashgrid:
                        min_logres: int = 5
                        max_logres: int = 11
                        dict_size: int = 19  # log2_hashmap_size
                        dim: int = 2  # n_features_per_level
                        range: list = [-0.5, 0.5]

                    hashgrid = Hashgrid()

                    class Coarse2Fine:
                        enabled: bool = True
                        init_active_level: int = 4
                        step: int = 1000
                        start_step: int = 5000

                    coarse2fine = Coarse2Fine()

                encoding = Encoding()

                class Gradient:
                    taps: int = 4  # 4 or 6
                    type: str = "finite_difference"  # "analytic" or "finite_difference"

                gradient = Gradient()

                feature_dim: int = 65

            sdf = SDF()

            class RGB:
                class MLP:
                    num_layers: int = 2
                    hidden_dim: int = 64
                    skip: list = []
                    activ: str = "relu"
                    weight_norm: bool = True

                mlp = MLP()
                mode: str = "idr"  # "idr" or "nerf"

                class EncodingView:
                    type: str = "spherical"  # "spherical" or "fourier"
                    levels: int = 4  # SH degree

                encoding_view = EncodingView()

            rgb = RGB()

            class SVar:
                init_val: float = 0.3
                anneal_end: float = 0.1

            s_var = SVar()

        object = Object()

        class Background:
            enabled: bool = True
            white: bool = True

            class MLP:
                num_layers: int = 1
                hidden_dim: int = 64

            mlp = MLP()

            class Encoding:
                levels: int = 16

                class Hashgrid:
                    dict_size: int = 19
                    dim: int = 2

                hashgrid = Hashgrid()

            encoding = Encoding()

        background = Background()

        class AppearEmbed:
            enabled: bool = False
            dim: int = 16

        appear_embed = AppearEmbed()

    model = Model()

    # ==================== NerfAcc 加速配置 ====================
    class NerfAcc:
        enabled: bool = True

        class OccGrid:
            resolution: int = 128
            resolution_bg: int = 256
            occ_thre: float = 0.001
            occ_thre_bg: float = 0.01

        occ_grid = OccGrid()

        grid_prune: bool = True
        dynamic_ray_sampling: bool = True
        max_train_num_rays: int = 8192

    nerfacc = NerfAcc()

    # ==================== 检查点配置 ====================
    class Checkpoint:
        strict_resume: bool = True
        save_every_n_epochs: int = 1

    checkpoint = Checkpoint()

    # ==================== 导出配置 ====================
    class Export:
        resolution: int = 512
        block_res: int = 64
        chunk_size: int = 2097152

    export = Export()

    # ==================== 其他配置 ====================
    logdir: str = "./exp"
    max_epoch: int = 20
    seed: int = 42

    @property
    def iters_per_epoch(self):
        return self.trainer.max_steps // self.max_epoch


def load_config_from_yaml(yaml_path: str) -> Config:
    """从 YAML 文件加载配置"""
    import yaml

    with open(yaml_path, 'r') as f:
        yaml_config = yaml.safe_load(f)

    config = Config()
    _update_config_from_dict(config, yaml_config)
    return config


def _update_config_from_dict(config_obj, config_dict: dict, prefix: str = ""):
    """递归更新配置对象"""
    for key, value in config_dict.items():
        if hasattr(config_obj, key):
            attr = getattr(config_obj, key)
            if isinstance(value, dict) and hasattr(attr, '__dict__'):
                _update_config_from_dict(attr, value, f"{prefix}.{key}")
            else:
                setattr(config_obj, key, value)
