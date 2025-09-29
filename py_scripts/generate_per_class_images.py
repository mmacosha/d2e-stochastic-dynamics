from pathlib import Path
import click

import torch
import torch.nn as nn

from torchvision.utils import save_image

from sb.data.datasets import ClsRewardDist 


@click.command()
@click.option("--gen_name",     "gen_name",             type=click.STRING)
@click.option("--cls_name",     "cls_name",             type=click.STRING)
@click.option("--save_path",    "save_path",            default="/workspace/writeable/gen_images", type=click.STRING)
@click.option("--num_classes",  "num_of_classes",       default=10,     type=click.INT)
@click.option("--max_num",      "max_num_per_class",    default=10_000, type=click.INT)
@click.option("--batch_size",   "batch_size",           default=1024,   type=click.INT)
@click.option("--device",       "device",               default="cpu",  type=click.STRING)
@click.option("--dim",          "dim",                  default=512,    type=click.INT)
@click.option("--sample",       "sample",               is_flag=True)
@torch.no_grad()
def generate_per_class_images(
        gen_name: str,
        cls_name: str,
        save_path: str, 
        num_of_classes: int = 10,
        max_num_per_class: int = 10_000,
        batch_size: int = 1024,
        device: str = 'cpu',
        dim: int = 512,
        sample: bool = False,
    ):
    save_path = Path(save_path) / (f"{gen_name}-sample" if sample else gen_name)
    device = device if device in {"mps", "cpu"} else f"cuda:{device}"
    save_path.mkdir(exist_ok=False)

    num_of_images = torch.tensor([0] * num_of_classes)
    
    print("Initialising generator and classifier ... ")
    reward = ClsRewardDist(
        gen_name, cls_name, dim, 
        "/workspace/writeable", [0], 
        'sum', device
    ).reward
    
    print("Creating directories for images ... ")
    for cls_ in range(num_of_classes):
        (save_path / f"class_{cls_}").mkdir(exist_ok=False)

    print("Generating images ... ")
    it = 0
    while torch.any(num_of_images < max_num_per_class):
        z = torch.randn(batch_size, dim, device=device)
        output = reward(z, beta=0.1)
        if sample:
            images, probs = output["images"], output["all_probas"]
            cls_ids = torch.multinomial(
                probs, num_samples=1, replacement=True
            ).squeeze(1)
        else:
            images, cls_ids = output["images"], output["classes"]

        for img, cls_id in zip(output["images"], cls_ids):
            curr_img_id = num_of_images[cls_id]

            if curr_img_id < max_num_per_class:
                img_name = f"cls_{cls_id:02d}_img_{curr_img_id:05d}.png"
                save_image(img, save_path / f"class_{cls_id}" / img_name)
                num_of_images[cls_id] += 1

        if it > 0 and it % 10 == 0:
            progress = " | ".join(
                f"cls {i}: {int(100 * num_of_images[i] / max_num_per_class):03d}%" 
                for i, n in enumerate(num_of_images)
            )
            print(f"Progress: {progress}")
        
        it += 1
    
    print("Done!")


if __name__ == "__main__":
    generate_per_class_images()
