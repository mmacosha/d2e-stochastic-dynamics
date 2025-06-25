#!/bin/bash
python train.py --cfg=ablate --name=sb-normal_to_gauss-match_ll-train_log_var        --wandb=online --overrides='data.p0=d2normal,data.p1=8gaussians,sampler.matching_method=ll,sampler.train_log_var=true' && \
python train.py --cfg=ablate --name=sb-normal_to_8gauss-match_ll-no_train_log_var    --wandb=online --overrides='data.p0=d2normal,data.p1=8gaussians,sampler.matching_method=ll,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-normal_to_8gauss-match_mean-no_train_log_var  --wandb=online --overrides='data.p0=d2normal,data.p1=8gaussians,sampler.matching_method=mean,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-normal_to_8gauss-match_score-no_train_log_var --wandb=online --overrides='data.p0=d2normal,data.p1=8gaussians,sampler.matching_method=score,sampler.train_log_var=false' && \

python train.py --cfg=ablate --name=sb-normal_to_moons-match_ll-train_log_var       --wandb=online --overrides='data.p0=d2normal,data.p1=moons,sampler.matching_method=ll,sampler.train_log_var=true' && \
python train.py --cfg=ablate --name=sb-normal_to_moons-match_ll-no_train_log_var    --wandb=online --overrides='data.p0=d2normal,data.p1=moons,sampler.matching_method=ll,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-normal_to_moons-match_mean-no_train_log_var  --wandb=online --overrides='data.p0=d2normal,data.p1=moons,sampler.matching_method=mean,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-normal_to_moons-match_score-no_train_log_var --wandb=online --overrides='data.p0=d2normal,data.p1=moons,sampler.matching_method=score,sampler.train_log_var=false' && \

python train.py --cfg=ablate --name=sb-moons_to_8gauss-match_ll-train_log_var       --wandb=online --overrides='data.p0=moons,data.p1=8gaussians,sampler.matching_method=ll,sampler.train_log_var=true' && \
python train.py --cfg=ablate --name=sb-moons_to_8gauss-match_ll-no_train_log_var    --wandb=online --overrides='data.p0=moons,data.p1=8gaussians,sampler.matching_method=ll,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-moons_to_8gauss-match_mean-no_train_log_var  --wandb=online --overrides='data.p0=moons,data.p1=8gaussians,sampler.matching_method=mean,sampler.train_log_var=false' && \
python train.py --cfg=ablate --name=sb-moons_to_8gauss-match_score-no_train_log_var --wandb=online --overrides='data.p0=moons,data.p1=8gaussians,sampler.matching_method=score,sampler.train_log_var=false'
