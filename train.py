import hydra
import omegaconf

from samplers import SBTrainer, SBConfig
from datasets_2d import DatasetSampler
from model import SimpleNet


@hydra.main(version_base=None, config_path="./configs",  config_name="config")
def run(config: omegaconf.DictConfig):
    sampler = DatasetSampler(
        p_0=config.data.p_0.name, p_0_args=config.data.p_0.args,
        p_1=config.data.p_1.name, p_1_args=config.data.p_1.args
    )
    fwd_model = SimpleNet(**config.models.fwd)
    bwd_model = SimpleNet(**config.models.bwd)

    sb_config = SBConfig(**config.sampler)

    sb_trainer = SBTrainer(
        F=fwd_model,
        B=bwd_model,
        sampler=sampler,
        config=sb_config,
        wandb_config=config.exp,
    )
    sb_trainer.train()


if __name__ == "__main__":
    run()