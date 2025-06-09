import click
from hydra import initialize, compose
from omegaconf import OmegaConf

@click.command()
@click.option('--cfg_path', 'cfg_path', type=click.Path(exists=True), default='configs')
@click.option("--cfg", 'cfg', type=click.STRING, default='config')
def main(cfg_path, cfg):
    overrides = []

    with initialize(version_base=None, config_path=cfg_path):
        cfg = compose(config_name=cfg, overrides=overrides)

    # Use the loaded config
    print("Loaded config:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()