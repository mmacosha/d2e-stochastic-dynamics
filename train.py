import os
import random

import torch
import numpy as np

import click
import regex as re
from hydra import initialize, compose
from omegaconf import OmegaConf

from sb.data import datasets
from sb.nn import SimpleNet, DSFixedBackward
from sb.samplers import (
    SBConfig, D2ESBConfig,
    D2DSB, D2ESB_2D, D2ESB_IMG, D2DSBLangevin
)


def read_overrides(overrides):
    if not overrides:
        return []
    pattern = r"[\w@.]+=(?:\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[^,]+)"
    return re.findall(pattern, overrides)


def seed_everything(seed: int = 42):
    """
    Set random seeds for reproducibility across Python, NumPy, and PyTorch (CPU & CUDA).

    Args:
        seed (int): Random seed to use.
    """
    # Python's built-in RNG
    random.seed(seed)

    # NumPy RNG
    np.random.seed(seed)

    # PyTorch RNGs
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Force deterministic algorithms (will throw if unsupported ops are used)
    torch.use_deterministic_algorithms(True)

    # Environment variables for reproducibility
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # required for reproducibility with cuBLAS

    print(f"Seed set to {seed} (deterministic mode ON)")

# Example usage:
# seed_everything(123)


@click.command()
@click.option('--cfg_path',     "cfg_path",  type=click.Path(exists=True), default='configs')
@click.option("--cfg",          "cfg",       type=click.STRING, default='config')
@click.option("--name",         "name",      type=click.STRING, default=None)
@click.option("--run_id",       "run_id",    type=click.STRING, default=None)
@click.option("--wandb",        "wandb",     type=click.STRING, default='online')
@click.option("--device",       "device",    type=click.STRING, default=0)
@click.option("--seed",         "seed",      type=click.INT, default=42 )
@click.option("--debug",        "debug",     type=click.BOOL, default=False, is_flag=True)
@click.option("--overrides",    "overrides", type=click.STRING, default=None,)
def main(cfg_path: str, cfg: str, name: str, run_id: str,  wandb: str, 
         device: int, seed: int, debug: bool, overrides=None):
    seed_everything(seed)
    with initialize(version_base=None, config_path=cfg_path):
        overrides = read_overrides(overrides)
        
        print('\nThe following overrides are applied:')
        for override in overrides:
            print("    ", override)
        print(flush=True)

        OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
        config = compose(config_name=cfg, overrides=overrides)

        if debug:
            print("\nATTENTION: DEBUG MODE IS ON!\n")
            print("SOME PARAMETERS CAN:")
            config.exp.mode = 'disabled'
            config.exp.name = f'debug-run-{config.exp.name}'
            config.sampler.num_fwd_steps=10
            config.sampler.num_bwd_steps=10
            config.sampler.num_sb_steps = 2
        else:
            config.exp.id = run_id 
            config.exp.mode = wandb
            config.exp.name = f"{name if name else config.exp.name}-{seed=}"
        
        config.sampler.device = device if device == 'mps' else f"cuda:{device}"

    if config.sampler.matching_method not in  {'ll', 'sf2m'} and \
       (config.models.fwd.predict_log_var or config.models.bwd.predict_log_var):
        raise ValueError(
            f"Matching method {config.sampler.matching_method} " \
            "do not support tainable variance."
        )
    
    p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
    p1 = datasets[config.data.p_1.name](**config.data.p_1.args)

    fwd_model = SimpleNet(**config.models.fwd).to(config.sampler.device)
    bwd_model = SimpleNet(**config.models.bwd).to(config.sampler.device)

    match config.sampler.name:
        case'd2d':
            sb_config = SBConfig(**config.sampler)
            sb_trainer = D2DSB(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p0=p0, p1=p1,
                config=sb_config,
            )
        case 'd2e_2d':
            sb_config = D2ESBConfig(**config.sampler)
            sb_trainer = D2ESB_2D(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p0=p0, p1=p1,
                config=sb_config,
                buffer_config=config.buffer
            )

        case 'd2e':
            sb_config = D2ESBConfig(**config.sampler)
            sb_trainer = D2ESB_IMG(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p0=p0, p1=p1,
                config=sb_config,
                buffer_config=config.buffer
            )
        case 'd2d_langevin':
            sb_config = SBConfig(**config.sampler)
            sb_trainer = D2DSBLangevin(
                fwd_model=fwd_model,
                bwd_model=bwd_model,
                p0=p0, p1=p1,
                config=sb_config,
                buffer_config=config.buffer
            )
        case _: 
            raise NotImplementedError('this trainer is not available')
    
    sb_trainer.train(config.exp, full_cfg=config)


if __name__ == "__main__":
    main()
