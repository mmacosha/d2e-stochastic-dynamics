import click
from omegaconf import OmegaConf
from hydra import initialize, compose


@click.command()
@click.option('--cfg_path', 'cfg_path', type=click.Path(exists=True), default='configs')
@click.option("--cfg", 'cfg', type=click.STRING, default='config')
@click.option("--overrides", 'overrides', type=click.STRING, default=None,)
def main(cfg_path: str, cfg: str, overrides=None):
    with initialize(version_base=None, config_path=cfg_path):
        overrides = overrides.split(',') or []
        config = compose(config_name=cfg, overrides=overrides)
    
    print(OmegaConf.to_yaml(config, resolve=True))


if __name__ == "__main__":
    main()
