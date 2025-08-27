from pathlib import Path

import click
import regex as re
from datetime import datetime
from omegaconf import OmegaConf

from tqdm import trange

import torch
import torchvision

from sb.nn import SimpleNet
from sb.data import datasets
from sb.nn.reward import ClsReward
from sb.samplers import utils
from sb.samplers.base_class import find_checkpoint

import wandb


def read_overrides(overrides):
    if not overrides:
        return []
    pattern = r"[\w@.]+=(?:\"[^\"]*\"|'[^']*'|\[[^\]]*\]|[^,]+)"
    return re.findall(pattern, overrides)


@click.command()
@click.option("--device",       "device",       type=click.INT,     default=0)
@click.option("--batch_size",   "batch_size",   type=click.INT,     default=64)
@click.option("--num_batches",  "num_batches",  type=click.INT,     default=10)
@click.option("--run_id",       "run_id",       type=click.STRING,  default=None)
def generate(
    device: int,
    batch_size: int,
    num_batches: int,
    run_id: str = None,
    ):
    device = f"cuda:{device}"
    run_path = [*Path("/workspace/writeable/wandb").glob(f"*{run_id}*")][-1]

    OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
    config = OmegaConf.load(run_path / "files" / "full_run_config.yaml")
    config.data.p_0.args.device = device

    p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
    reward = ClsReward.build_reward(
        config.data.p_1.args.generator_type, 
        config.data.p_1.args.classifier_type, 
        config.data.p_1.args.target_classes, 
        config.data.p_1.args.reward_type, 
        config.data.p_1.args.reward_dir
    )
    reward.to(device).eval()
    
    fwd_model = SimpleNet(**config.models.fwd)
    wandb_dir = Path("/workspace/writeable/wandb")
    checkpoint_path, _ = find_checkpoint(wandb_dir, -1, run_id)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    fwd_model.load_state_dict(checkpoint['forward'])
    fwd_model.to(device).eval()

    wandb.init(project="sb-gen")
    with torch.no_grad():
        for _ in trange(num_batches, desc="Generating images", unit="batch"):
            x0 = p0.sample(batch_size).to(device)
            x1_pred = utils.sample_trajectory(
                fwd_model, x0, 'forward', 
                config.sampler.dt, config.sampler.n_steps, config.sampler.t_max, 
                only_last=True
            )
            prior_images = reward.generator(x0).cpu() * 0.5 + 0.5
            
            prior_images = torchvision.utils.make_grid(
                prior_images.clip(0, 1), nrow=2, normalize=True, 
                padding=0, pad_value=1.0
            )
            posterior_images = torchvision.utils.make_grid(
                reward(x1_pred)["images"].clip(0, 1).cpu(), 
                nrow=2, normalize=True, 
                padding=0, pad_value=1.0
            )
            wandb.log({
                "prior": wandb.Image(prior_images),
                "posterior": wandb.Image(posterior_images),
            })
    print("Generation complete!")


if __name__ == "__main__":
    generate()
