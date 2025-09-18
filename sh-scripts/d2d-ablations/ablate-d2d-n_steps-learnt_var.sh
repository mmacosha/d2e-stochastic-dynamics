#!/bin/bash

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

# train var
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=5 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=5 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=5 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=5 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=5 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'

train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=10 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=10 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=10 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=10 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'

train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-data2data-learnt_var-n_steps_=40 --wandb=online \
#    --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'