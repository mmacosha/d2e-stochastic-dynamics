import hydra
import omegaconf

from samplers import SBConfig, D2DSB, D2ESB, D2ESBConfig
from model import SimpleNet

from data import datasets


@hydra.main(version_base=None, config_path="./configs",  config_name="config")
def run(config: omegaconf.DictConfig):
    p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
    p1 = datasets[config.data.p_1.name](**config.data.p_1.args)

    fwd_model = SimpleNet(**config.models.fwd)
    bwd_model = SimpleNet(**config.models.bwd)

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