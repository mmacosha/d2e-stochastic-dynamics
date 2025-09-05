#!/bin/bash
# mean 
# python train.py --seed=42 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_mean --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# python train.py --seed=43 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_mean --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# python train.py --seed=44 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_mean --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean'

# # score
# python train.py --seed=42 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_score --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# python train.py --seed=43 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_score --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# python train.py --seed=44 --device=1 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_score --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score'

# mean 
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_mean --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_mean --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_mean --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean'

# score
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_score --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_score --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_score --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score'