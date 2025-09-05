#!/bin/bash
python train.py --seed=42 --device=mps --cfg=d2e-2d-no-langevin --name=sb-normal-to-gmm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
python train.py --seed=43 --device=mps --cfg=d2e-2d-no-langevin --name=sb-normal-to-gmm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
python train.py --seed=44 --device=mps --cfg=d2e-2d-no-langevin --name=sb-normal-to-gmm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'