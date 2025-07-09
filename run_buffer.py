import click
import wandb
import torch

import matplotlib.pyplot as plt

from sb.buffers import DecoupledLangevinBuffer
from sb.data.datasets import ClsRewardDist
from sb import utils


def read_classes(ctx, param, values):
    if not values:
        raise ValueError("At least one class should be provided.")

    values = values.split(",")
    return [int(v) for v in values if v != '']


@click.command()
@click.option("--device",       "device",               default=0,          type=click.INT)
@click.option("--gen",          "gen",                  default="stylegan", type=click.STRING)
@click.option("--cls",          "cls_",                 default="vgg",      type=click.STRING)
@click.option("--classes",      "classes",              default="0,",        callback=read_classes)
@click.option("--mode",         "mode",                 default="online",   type=click.STRING)
@click.option("--dim",          "dim",                  default=512,        type=click.INT)
@click.option("--buffer_size",  "buffer_size",          default=128,        type=click.INT)
@click.option("--lambda",       "ema_lambda",           default=0.0,        type=click.FLOAT)
@click.option("--step_size",    "init_step_size",       default=0.01,       type=click.FLOAT)
@click.option("--anneal",       "anneal_value",         default=0.1,        type=click.FLOAT)
@click.option("--steps",        "num_langevin_steps",   default=500,        type=click.INT)
@click.option("--plot_size",    "plot_size",            default=36,         type=click.INT)
@click.option("--img_shape",    "img_shape",            default=32,         type=click.INT)
def main(device: int, gen: str, cls_: str, classes, mode: str, 
         dim: int,  buffer_size: int,ema_lambda: float, 
         init_step_size: float,  anneal_value: float, 
         num_langevin_steps: int, plot_size: int,  img_shape: int):
    
    config = {
        "device": device,
        "gen": gen,
        "cls_": cls_,
        "classes": classes,
        "mode": mode,
        "dim": dim,
        "buffer_size": buffer_size,
        "ema_lambda": ema_lambda,
        "init_step_size": init_step_size,
        "anneal_value": anneal_value,
        "num_langevin_steps": num_langevin_steps,
        "plot_size": plot_size,
        "img_shape": img_shape,        
    }

    device = torch.device(f"cuda:{device}")
    dist = ClsRewardDist(
        gen, cls_, dim, "/workspace/writeable", classes, 'sum', device
    )
    
    lb = DecoupledLangevinBuffer(
        langevin_freq=1,
        buffer_size=buffer_size,
        p1=dist,
        init_step_size=init_step_size,
        num_langevin_steps=num_langevin_steps,
        ema_lambda=ema_lambda,
        sampler='ula2',
        noise_start_ratio=0.5,
        anneal_value=anneal_value,
        device=device,
        beta_fn="lambda i: 0.1"
    )
    name = "--".join([
        f"{gen=}",
        f"{cls_=}",
        f"num_steps={num_langevin_steps}",
        f"step_size={init_step_size}",
        f"annealing={anneal_value}",
        f"ema={ema_lambda}"
    ])

    with wandb.init(project='check-sampler', name=name, mode=mode, config=config):
        lb.run_langevin(dim=dim)
        with torch.no_grad():
            latent = lb.sample(buffer_size)
            output = dist.reward(latent)
            target_img, target_probas, target_cls = dist.reward.get_target_class_images(
                output, size=plot_size
            )
            target_img, target_probas, target_cls = (
                target_img.cpu(), target_probas.cpu(), target_cls.cpu()
            )
            if target_img.size(0) < plot_size:
                size = target_img.size(0)
                target_img = torch.cat([
                    target_img, torch.zeros(plot_size - size, 3, img_shape, img_shape)
                ])
                target_probas = torch.cat([
                    target_probas, torch.zeros(plot_size - size)
                ])
                target_cls = torch.cat([target_cls, torch.zeros(plot_size - size)])

        img, probas, classes = output["images"], output["probas"], output["classes"]
        random_img = utils.plot_annotated_images(
                img[:plot_size].clip(0, 1).cpu().view(-1, 3, img_shape, img_shape),
                (probas[:plot_size], classes[:plot_size]), 
                n_col=4, figsize=(18, 18)
            )
        target_cls_images = utils.plot_annotated_images(
            target_img.clip(0, 1).cpu().view(-1, 3, img_shape, img_shape), 
            (target_probas, target_cls), n_col=4, figsize=(18, 18)
        )
        wandb.log({
            "buffer_samples": wandb.Image(random_img),
            "target_class_buffer_samples": wandb.Image(target_cls_images),
        })
        plt.close("all")


if __name__ == "__main__":
    main()
