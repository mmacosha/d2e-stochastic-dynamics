#!/bin/bash

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

train --seed=42 --device=cpu --cfg=d2e-ablate --wandb=online --name=d2d-40GMM-dt_1-normal_std_20 \
    --overrides="data.p_0.args.std=20,data@data.p_1=40gmm,sampler.plot_limits=[-40, 40],sampler.dt=1,buffer=simple" && \
train --seed=43 --device=cpu --cfg=d2e-ablate --wandb=online --name=d2d-40GMM-dt_1-normal_std_20 \
    --overrides="data.p_0.args.std=20,data@data.p_1=40gmm,sampler.plot_limits=[-40, 40],sampler.dt=1,buffer=simple" && \
train --seed=44 --device=cpu --cfg=d2e-ablate --wandb=online --name=d2d-40GMM-dt_1-normal_std_20 \
    --overrides="data.p_0.args.std=20,data@data.p_1=40gmm,sampler.plot_limits=[-40, 40],sampler.dt=1,buffer=simple" && \
train --seed=45 --device=cpu --cfg=d2e-ablate --wandb=online --name=d2d-40GMM-dt_1-normal_std_20 \
    --overrides="data.p_0.args.std=20,data@data.p_1=40gmm,sampler.plot_limits=[-40, 40],sampler.dt=1,buffer=simple" && \
train --seed=46 --device=cpu --cfg=d2e-ablate --wandb=online --name=d2d-40GMM-dt_1-normal_std_20 \
    --overrides="data.p_0.args.std=20,data@data.p_1=40gmm,sampler.plot_limits=[-40, 40],sampler.dt=1,buffer=simple"

#  Reuse bwd traj 
# train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_reuse_bwd_trajectory --wandb=online \
#     --overrides="sampler.reuse_bwd_trajectory=True"

# no langevin
# train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_simple_buffer-do-not-reuse_bwd_traj --wandb=online \
#     --overrides="buffer=simple,sampler.reuse_bwd_trajectory=False"

# on-policy
# train --seed=42 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=43 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=44 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=45 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
# train --seed=46 --device=mps --cfg=d2e-ablate --name=sb-normal-to-gmm-ablation_on-policy --wandb=online \
#     --overrides="buffer=simple,sampler.off_policy_fraction=0.0,sampler.reuse_bwd_trajectory=False"
