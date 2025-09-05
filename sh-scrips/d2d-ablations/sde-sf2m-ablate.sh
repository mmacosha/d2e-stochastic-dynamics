#!/bin/bash
# # SDE
# python train.py --seed=42 --device=3 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# python train.py --seed=43 --device=3 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# python train.py --seed=44 --device=3 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde'

# SF2M
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_8gauss-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_8gauss-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_8gauss-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000'

# SDE
#python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
#python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
#python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_sde --wandb=online \
#   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde'

# SF2M
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_scurve-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_scurve-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-3-sb-normal_to_scurve-match_sf2m --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=sf2m,sampler.num_bwd_steps=0,sampler.num_fwd_steps=8000'
