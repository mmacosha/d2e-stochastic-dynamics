from pathlib import Path

import torch
import click
import wandb
from tqdm.auto import tqdm

from omegaconf import OmegaConf

from sb import metrics
from sb import nn as sbnn
from sb.data import datasets


def get_checkpoint_from_run_id(wandb_path, run_id):
    run_path = [*Path(wandb_path).glob(f"*{run_id}*")][-1]
    ckpt_path = run_path / "files" / "checkpoints" / "checkpoint-final.pth"
    config_path = run_path / "files" / "full_run_config.yaml"
    
    ckpt = torch.load(ckpt_path)
    cfg = OmegaConf.load(config_path)
    
    return ckpt, cfg


def recompute_kl(wandb_path, run_ids, device):
    OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
    kl_values = {}

    for run_id in (pbar := tqdm(run_ids, leave=False)):
        pbar.set_description(f"Processing {run_id}...")
        ckpt, config = get_checkpoint_from_run_id(wandb_path, run_id)
        
        fwd_model = sbnn.SimpleNet(**config.models.fwd)
        fwd_model.load_state_dict(ckpt['forward'])
        fwd_model.to(config.sampler.device)
        fwd_model.eval()

        p0 = datasets[config.data.p_0.name](**config.data.p_0.args)
        x0 = p0.sample(10_000).float().to(device)

        dt = config.sampler.dt
        t_max = config.sampler.t_max
        n_steps = config.sampler.n_steps
        alpha = config.sampler.alpha
        var = config.sampler.var
        method = config.sampler.matching_method

        path_kl = metrics.compute_path_kl(
            fwd_model, 
            x0, dt, t_max, n_steps, alpha, var, 
            method=method
        )
        name = f"{config.sampler.name}-{config.exp.name}"
        kl_values[name] = path_kl.item()

    with wandb.init(project="recomputed_path_kl", name="recomputed_path_kl"):
        for name, value in kl_values.items():
            wandb.run.summary[name.strip()] = value

@click.command()
@click.option('--wandb_path',   type=str, default='./wandb',    required=True, help='Path to the wandb directory containing runs.')
@click.option('--device',       type=str, default='mps',        help='Device to use for computation.')
@click.option('--run_ids',      type=str,                       required=True, help='Comma-separated list of run IDs to process.')
def main(wandb_path, device, run_ids):
    run_ids_list = run_ids.split(',')
    recompute_kl(wandb_path, run_ids_list, device)


if __name__ == "__main__":
    main()
