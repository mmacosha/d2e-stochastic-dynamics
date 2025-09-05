#!/bin/bash
# # DSBM
# python train.py --seed=42 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_dsbm --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# python train.py --seed=43 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_dsbm --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# python train.py --seed=44 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_dsbm --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm'

# # DSBM
# python train.py --seed=42 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_dsbm++ --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# python train.py --seed=43 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_dsbm++ --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# python train.py --seed=44 --device=2 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_dsbm++ --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++'

# DSBM
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm'

# DSBM
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm++ --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm++ --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_dsbm++ --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++'
