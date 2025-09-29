import torch
import torchvision
import numpy as np
from PIL import Image
import requests
import os
import wandb # Import Weights & Biases

import sys, os
sys.path.append(os.path.abspath("./external/sg3"))
import dnnlib, legacy

# --- Configuration ---
project_name = "StyleGAN2-ADA_CIFAR10_Generation"
run_name = "unconditional_generation_example"
output_dir = '/workspace/writeable/generated_images_wandb'
os.makedirs(output_dir, exist_ok=True)

num_images_to_generate = 32 # Number of images to generate and log
truncation_psi = 0.7       # Controls diversity vs. quality (0.5-1.0 common)
seed = 42                  # For reproducibility of latent vectors
save_local_grid = True     # Whether to save a local PNG grid of images

# --- 1. Initialize Weights & Biases ---
wandb.init(project=project_name, name=run_name, config={
    "num_images": num_images_to_generate,
    "truncation_psi": truncation_psi,
    "seed": seed,
    "model": "stylegan2_cifar10",
    "noise_mode": "random" # Default for generation
})

# --- 2. Set up device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
wandb.config.update({"device": str(device)})


# --- 3. Load the pre-trained StyleGAN2-ADA model for CIFAR-10 ---
try:
    print("Loading StyleGAN2-ADA model for CIFAR-10...")
    with dnnlib.util.open_url('/workspace/writeable/rewards/sg/stylegan2-cifar10-32x32.pkl') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    print("Model loaded successfully.")
except requests.exceptions.HTTPError as e:
    print(f"Failed to load model. Please check your internet connection or the repository URL. Error: {e}")
    wandb.alert(
        title="Model Loading Failed",
        text=f"Failed to load StyleGAN2-ADA model. Error: {e}",
        level=wandb.AlertLevel.ERROR
    )
    wandb.finish()
    exit()

# Set the model to evaluation mode
G.eval()

# --- 4. Generate Latent Vectors ---
torch.manual_seed(seed)
latents = torch.randn([num_images_to_generate, G.z_dim]).to(device)
print(f"Generated {num_images_to_generate} latent vectors of shape: {latents.shape}")

# --- 5. Generate Images ---
print(f"Generating {num_images_to_generate} images...")
with torch.no_grad():
    c = torch.zeros(
            (latents.shape[0], G.c_dim), 
            device=latents.device)
    generated_images_tensor = G(latents, c=c, truncation_psi=truncation_psi)

# --- 6. Post-process Images for Saving/Logging ---
# Convert from [-1, 1] to [0, 1]
generated_images_tensor = torch.clamp(generated_images_tensor * 0.5 + 0.5, 0, 1)

# Convert to a list of PIL Images for W&B logging
pil_images = []
for i in range(num_images_to_generate):
    # Permute from (C, H, W) to (H, W, C) for numpy, then to PIL
    img_np = generated_images_tensor[i].cpu().numpy().transpose(1, 2, 0)
    img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
    pil_images.append(img_pil)

# --- 7. Log Images to Weights & Biases ---
print("Logging images to Weights & Biases...")
# Create a W&B table to log images with additional info (optional)
columns = ["ID", "Generated Image", "Truncation Psi", "Seed"]
data = []
for i, img_pil in enumerate(pil_images):
    data.append([i, wandb.Image(img_pil), truncation_psi, seed])

wandb.log({"Generated CIFAR-10 Images": wandb.Table(columns=columns, data=data)})
print("Images logged to W&B successfully!")

# --- 8. Optionally, save a local grid of images ---
if save_local_grid:
    output_path = os.path.join(output_dir, 'cifar10_stylegan_grid_wandb.png')
    # Use torchvision.utils.save_image for convenience with a tensor
    torchvision.utils.save_image(generated_images_tensor, output_path, nrow=int(np.sqrt(num_images_to_generate)))
    print(f"Local image grid saved to '{output_path}'")

# --- 9. Finish W&B Run ---
wandb.finish()
print("W&B run finished.")