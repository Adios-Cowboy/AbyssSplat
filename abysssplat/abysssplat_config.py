"""Nerfstudio method configuration for AbyssSplat."""

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.plugins.types import MethodSpecification

from abysssplat.abysssplat import AbyssSplatModelConfig


NUM_STEPS = 15_000


def _exponential_optimizer(lr: float, lr_final: float) -> dict:
    return {
        "optimizer": AdamOptimizerConfig(lr=lr, eps=1e-15),
        "scheduler": ExponentialDecaySchedulerConfig(
            lr_final=lr_final,
            max_steps=NUM_STEPS,
        ),
    }


abysssplat_method = MethodSpecification(
    config=TrainerConfig(
        method_name="abysssplat",
        steps_per_eval_image=1_000,
        steps_per_eval_batch=0,
        steps_per_save=2_000,
        steps_per_eval_all_images=1_000,
        max_num_iterations=NUM_STEPS,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
            ),
            model=AbyssSplatModelConfig(
                num_steps=NUM_STEPS,
                main_loss="reg_l1",
                ssim_loss="reg_ssim",
                zero_medium=False,
            ),
        ),
        optimizers={
            "means": _exponential_optimizer(1.6e-4, 5e-5),
            "features_dc": _exponential_optimizer(0.0025, 0.0025),
            "features_rest": _exponential_optimizer(0.0025 / 20, 0.0025 / 20),
            "opacities": _exponential_optimizer(0.05, 0.05),
            "scales": _exponential_optimizer(0.005, 0.005),
            "quats": _exponential_optimizer(0.001, 0.001),
            "camera_opt": _exponential_optimizer(1e-3, 5e-5),
            "medium_mlp": _exponential_optimizer(1e-3, 1.5e-4),
            "direction_encoding": _exponential_optimizer(1e-3, 1.5e-4),
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="AbyssSplat for active-illumination-aware deep-sea reconstruction.",
)
