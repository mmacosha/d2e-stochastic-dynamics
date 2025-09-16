#!/bin/bash

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

DEVICE='cpu'

train --seed=42 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-base --wandb=online
# train --seed=43 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-base --wandb=online && \
# train --seed=44 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-base --wandb=online && \
# train --seed=45 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-base --wandb=online && \
# train --seed=46 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-base --wandb=online

#  Reuse bwd traj 
train --seed=42 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-reuse_bwd_trajectory --wandb=online \
    --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=43 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True" && \
# train --seed=44 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True" && \
# train --seed=45 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True" && \
# train --seed=46 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"

# no langevin
train --seed=42 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-simple_buffer-do-not-reuse_bwd_traj --wandb=online \
    --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=44 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=45 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=46 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"

# on-policy
train --seed=42 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-on-policy --wandb=online \
    --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=44 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=45 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"  && \
# train --seed=46 --device="$DEVICE" --cfg=d2e-ablate --name=sb-normal-to-gmm-on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
