#!/bin/bash

DEVICE=$1

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

# train var
train --seed=42 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=5 --wandb=online \
  --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
train --seed=43 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=5 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
train --seed=44 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=5 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
train --seed=45 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=5 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'
train --seed=46 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=5 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.04,sampler.n_steps=5'

train --seed=42 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
train --seed=43 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
train --seed=44 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
train --seed=45 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'
train --seed=46 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=10 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.02,sampler.n_steps=10'

train --seed=42 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=20 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.01,sampler.n_steps=20'
train --seed=43 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=20 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.01,sampler.n_steps=20'
train --seed=44 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=20 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.01,sampler.n_steps=20'
train --seed=45 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=20 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.01,sampler.n_steps=20'
train --seed=46 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=20 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.01,sampler.n_steps=20'

train --seed=42 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=43 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=44 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=45 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=46 --device=${DEVICE} --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
