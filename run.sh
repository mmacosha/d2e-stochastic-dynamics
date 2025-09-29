#!/bin/bash

train --seed=44 --device=0 --cfg=d2d-ablate --name=sb-data2data-fixed_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=45 --device=0 --cfg=d2d-ablate --name=sb-data2data-fixed_var-n_steps=40 --wandb=online  \
   --overrides='models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=46 --device=0 --cfg=d2e-ablate --name=fixed_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'
train --seed=46 --device=0 --cfg=d2e-ablate --name=learnt_var-n_steps=40 --wandb=online \
   --overrides='models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.dt=0.005,sampler.n_steps=40'