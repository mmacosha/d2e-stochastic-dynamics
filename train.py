import click
from hydra import initialize, compose

from sb.data import datasets
from sb.nn.mlp import SimpleNet
from sb.samplers import SBConfig, D2DSB, D2ESB, D2ESBConfig


@click.command()
@click.option('--cfg_path', 'cfg_path', type=click.Path(exists=True), default='configs')
@click.option("--cfg", 'cfg', type=click.STRING, default='config')
@click.option("--name", 'name', type=click.STRING, default=None)
@click.option("--wandb", 'wandb', type=click.STRING, default='online')
@click.option("--device", 'device', type=click.INT, default=0)
@click.option("--debug", 'debug', type=click.BOOL, default=False, is_flag=True)
@click.option("--overrides", 'overrides', type=click.STRING, default=None,)
def run(cfg_path: str, cfg: str, name: str, wandb: str, 
        device: int, debug: bool, overrides=None):
    with initialize(version_base=None, config_path=cfg_path):
        overrides = overrides.split(',') if overrides else []
        config = compose(config_name=cfg, overrides=overrides)
        if debug:
            print("\nATTENTION: DEBUG MODE IS ON!\n")
            config.exp.mode = 'disabled'
            config.exp.name = f'debug-run-{config.exp.name}'
            config.sampler.num_fwd_steps=10
            config.sampler.num_bwd_steps=10
            config.sampler.n_sb_iter = 2
        else:
            config.exp.mode = wandb
            config.exp.name = name if name else config.exp.name
        
        config.sampler.device = f"cuda:{device}"

    if config.sampler.matching_method in {"score", "mean"} and \
       (config.models.fwd.predict_log_var or config.models.bwd.predict_log_var):
        raise ValueError(
            "Matching method 'score' and 'mean' do not support tainable variance."
        )
    
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