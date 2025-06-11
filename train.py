import os

import click
import omegaconf
from hydra import initialize, compose

from samplers import SBConfig, D2DSB, D2ESB, D2ESBConfig
from models.simple_models import SimpleNet

from data import datasets


@click.command()
@click.option('--base_cfg_path', 'base_cfg_path', 
              type=click.Path(exists=True), default='configs')
@click.option("--cfg", 'cfg', type=click.STRING, default='config')
@click.option("--overrides", 'overrides', type=click.STRING, default=None,)
def run(base_cfg_path: str, cfg: str, overrides=None):
    with initialize(version_base=None, config_path=base_cfg_path):
        overrides = overrides.split(',') if overrides else []
        config = compose(config_name=cfg, overrides=overrides)
    
    p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
    p1 = datasets[config.data.p_1.name](**config.data.p_1.args)

    fwd_model = SimpleNet(**config.models.fwd).to(config.sampler.device)
    bwd_model = SimpleNet(**config.models.bwd).to(config.sampler.device)

    if  config.sampler.name == 'd2d':
        sb_config = SBConfig(**config.sampler)
        sb_trainer_cls = D2DSB
    elif config.sampler.name == 'd2e':
        sb_config = D2ESBConfig(**config.sampler)
        sb_trainer_cls = D2ESB
    else: 
        raise NotImplementedError('this trainer is not available')
    
    sb_trainer = sb_trainer_cls(
        fwd_model=fwd_model,
        bwd_model=bwd_model,
        p0=p0, p1=p1,
        config=sb_config,
    )
    sb_trainer.train(config.exp)


if __name__ == "__main__":
    run()