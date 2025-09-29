from pathlib import Path

import click
import regex as re
from datetime import datetime
from omegaconf import OmegaConf

from tqdm import trange

import torch
import torchvision


from sb import nn as sbnn
from sb import samplers
from sb import metrics
from sb import losses

from sb.data import datasets

import wandb


@click.command()
@click.option("--run_id",               "run_id",               type=click.STRING,  default=None)
@click.option("--batch_size",           "batch_size",           type=click.INT,     default=512)
@click.option("--num_images",           "num_images",           type=click.INT,     default=10_000)
@click.option("--device",               "device",               type=click.INT,     default=0)
@click.option("--save_path",            "save_path",            type=click.STRING,  default="/workspace/writeable/gen_images")
@click.option("--save_images",          "save_images",          is_flag=True)
@click.option("--log_images_to_wandb",  "log_images_to_wandb",  is_flag=True)
@click.option("--log_metrics",          "log_metrics",          is_flag=True)
def generate(
        run_id: str,
        device: int,
        batch_size: int,
        num_images: int,
        save_path: str,
        save_images: bool = False,
        log_images_to_wandb: bool = False,
        log_metrics: bool = False,
    ):

    device = f"cuda:{device}"
    run_path = max(
        Path("/workspace/writeable/wandb").glob(f"*{run_id}*"),
        key=lambda p: p.stat().st_mtime, default=None
    )

    OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
    config = OmegaConf.load(run_path / "files" / "full_run_config.yaml")
    config.data.p_0.args.device = device
    config.data.p_1.args.device = device

    dt = config.sampler.dt
    n_steps = config.sampler.n_steps
    t_max = config.sampler.t_max
    alpha = config.sampler.alpha
    var = 2 #config.sampler.var

    config.data.p_1.args.generator_type = 'cifar10-sngan'
    # classifier_type: cifar10-vgg13

    p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
    p1 = datasets[config.data.p_1.name](**config.data.p_1.args)

    # Load checkpoint
    wandb_dir = Path("/workspace/writeable/wandb")
    checkpoint_path, _ = samplers.base_class.find_checkpoint(wandb_dir, -1, run_id)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Restore fwd model
    fwd_model = sbnn.SimpleNet(**config.models.fwd)
    fwd_model.load_state_dict(checkpoint['forward'])
    fwd_model.to(device).eval()
    
    # Restore bwd model
    bwd_model = sbnn.SimpleNet(**config.models.bwd)
    bwd_model.load_state_dict(checkpoint['backward'])
    bwd_model.to(device).eval()

    curr_img_id = 0

    all_x0, all_x1 = [], []
    elbo_values = []
    path_kl_values = []
    log_reward_values = []

    if save_images:
        save_path = Path(save_path) / run_id
        save_path.mkdir(exist_ok=False)

    wandb.init(project="sb-gen", name=f"{config.exp.name}--run_id_{run_id}")
    with torch.no_grad():
        for _ in trange((num_images + batch_size - 1) // batch_size, 
                        desc="Generating images", unit="batch"):
            x0 = p0.sample(batch_size).to(device)
            all_x0.append(x0)
            
            x1_pred = samplers.utils.sample_trajectory(
                fwd_model,
                x0, dt, t_max, n_steps, alpha, var,
                direction="fwd",
                only_last=True,
                method=config.sampler.matching_method,
            )
            all_x1.append(x1_pred)

            prior_images = torch.clamp(p1.reward.generator(x0).cpu() * 0.5 + 0.5, 0, 1)
            outputs = p1.reward(x1_pred)
            posterior_images = outputs["images"].cpu()

            if log_metrics:
                path_kl = metrics.compute_path_kl(
                    fwd_model,
                    x0, dt, t_max, n_steps, alpha, var,
                    method=config.sampler.matching_method,
                    apply_averaging=False
                )
                elbo = losses.compute_fwd_tb_log_difference(
                    fwd_model, bwd_model, p1.log_density, 
                    x0, dt, t_max, n_steps,
                )
                elbo_values.append(elbo)
                path_kl_values.append(path_kl)
            
                reward, _ = p1.reward.get_reward_and_precision(outputs=outputs)
                log_reward_values.append(reward.log())


            if save_images:
                for img in posterior_images:
                    if curr_img_id == num_images:
                        break
                    img_name = f"run_{run_id}_posterior_img_{curr_img_id:05d}.png"
                    torchvision.utils.save_image(img, save_path / img_name)
                    curr_img_id += 1


            if log_images_to_wandb:            
                prior_images = torchvision.utils.make_grid(
                    prior_images,
                    nrow=8, normalize=True, 
                    padding=0, pad_value=1.0
                )
                posterior_images = torchvision.utils.make_grid(
                    posterior_images, 
                    nrow=8, normalize=True, 
                    padding=0, pad_value=1.0
                )
                wandb.log({
                    "prior": wandb.Image(prior_images),
                    "posterior": wandb.Image(posterior_images),
                })

    if log_metrics:
        x0 = torch.cat(all_x0, dim=0)[:num_images]
        x1 = torch.cat(all_x1, dim=0)[:num_images]
        wandb.run.summary[
            'total_x0_x1_transport_cost'
        ] = round((x0 - x1).pow(2).sum(1).mean().item(), 4)
        wandb.run.summary[
            'total_path_kl'
        ] = round(torch.cat(path_kl_values, dim=0)[:num_images].mean().item(), 4)
        wandb.run.summary[
            'total_elbo'
        ] = round(torch.cat(elbo_values, dim=0)[:num_images].mean().item(), 4)
        wandb.run.summary[
            'total_mean_log_reward'
        ] = round(torch.cat(log_reward_values, dim=0)[:num_images].mean().item(), 4)

    print("Generation complete!")


if __name__ == "__main__":
    generate()
