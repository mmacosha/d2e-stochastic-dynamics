#!/bin/bash
# train var
# python train.py --seed=42 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_ll-train_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
# python train.py --seed=43 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_ll-train_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
# python train.py --seed=44 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_gauss-match_ll-train_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'

# # fixed var
# python train.py --seed=42 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_ll-fixed_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
# python train.py --seed=43 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_ll-fixed_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
# python train.py --seed=44 --device=0 --cfg=ablate --name=TRY-2-sb-normal_to_8gauss-match_ll-fixed_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll'

# train var
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-train_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-train_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-train_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'

# fixed var
python train.py --seed=42 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-fixed_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
python train.py --seed=43 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-fixed_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
python train.py --seed=44 --device=mps --cfg=ablate --name=TRY-2-sb-normal_to_scurve-match_ll-fixed_log_var --wandb=online \
   --overrides='data@data.p_0=d2normal,data@data.p_1=scurve,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll'