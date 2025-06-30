#!/bin/bash
#python train.py --device=1 --cfg=ablate --name=sb-normal_to_gauss-match_ll-train_log_var        --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.train_log_var=true' && \
#python train.py --device=1 --cfg=ablate --name=sb-normal_to_8gauss-match_ll-no_train_log_var    --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.train_log_var=false' && \
#python train.py --device=1 --cfg=ablate --name=sb-normal_to_8gauss-match_mean-no_train_log_var  --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean,sampler.train_log_var=false' && \
#python train.py --device=1 --cfg=ablate --name=sb-normal_to_8gauss-match_score-no_train_log_var --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score,sampler.train_log_var=false' && \

python train.py --device=1 --cfg=ablate --name=sb-normal_to_moons-match_ll-train_log_var        --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.train_log_var=true' && \
python train.py --device=1 --cfg=ablate --name=sb-normal_to_moons-match_ll-no_train_log_var     --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.train_log_var=false' && \
python train.py --device=1 --cfg=ablate --name=sb-normal_to_moons-match_mean-no_train_log_var   --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean,sampler.train_log_var=false' && \
#python train.py --device=1 --cfg=ablate --name=sb-normal_to_moons-match_score-no_train_log_var  --wandb=online \
#    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score,sampler.train_log_var=false' && \

python train.py --device=1 --cfg=ablate --name=sb-moons_to_8gauss-match_ll-train_log_var        --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=8gaussians,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll,sampler.train_log_var=true' && \
python train.py --device=1 --cfg=ablate --name=sb-moons_to_8gauss-match_ll-no_train_log_var     --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll,sampler.train_log_var=false' && \
python train.py --device=1 --cfg=ablate --name=sb-moons_to_8gauss-match_mean-no_train_log_var   --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean,sampler.train_log_var=false'
#python train.py --device=1 --cfg=ablate --name=sb-moons_to_8gauss-match_score-no_train_log_var  --wandb=online \
#    --overrides='data@data.p_0=moons,data@data.p_1=8gaussians,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score,sampler.train_log_var=false'
