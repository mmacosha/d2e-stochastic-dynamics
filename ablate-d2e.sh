#!/bin/bash
# train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online && \
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online && \
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online && \
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online && \
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online

#  Reuse bwd traj 
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
    --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"

# no langevin
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
    --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"

# on-policy
train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
    --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
