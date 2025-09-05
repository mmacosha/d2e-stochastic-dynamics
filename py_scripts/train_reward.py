import click
from hydra import initialize, compose


from sb.training import (
    train_cifar_model, 
    # train_mnist_model
)
@click.command()
@click.option('--base_cfg_path', 'base_cfg_path', 
              type=click.Path(exists=True), default='configs/rewards')
@click.option("--cfg", 'cfg', type=click.STRING, default='config')
@click.option("--overrides", 'overrides', type=click.STRING, default=None,)
def run_training(base_cfg_path: str, cfg: str, overrides=None):
    with initialize(version_base=None, config_path=base_cfg_path):
        overrides = overrides.split(',') if overrides else []
        config = compose(config_name=cfg, overrides=overrides)

    if cfg == "cifar_cfg":
        train_cifar_model(config)
    else:
        raise NotImplementedError(f"Training for {cfg} is not implemented yet.")


if __name__ == "__main__":
    run_training()
