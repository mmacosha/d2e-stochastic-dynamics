#!/bin/bash
# ll matching learnt var
train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'

# ll matching fixed var
train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll'

# # ll score matching
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_score-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_score-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_score-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_score-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_score-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score'

# # ll mean matching
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_mean-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_mean-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_mean-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_mean-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_mean-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean'

# # ll dsbm
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm'

# # ll dsbm++
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm++-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm++-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm++-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm++-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_dsbm++-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++'

# # ll sf2m
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sf2m-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sf2m-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sf2m-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sf2m-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sf2m-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m'

# # ll sde
# train --seed=42 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sde-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# train --seed=43 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sde-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# train --seed=44 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sde-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# train --seed=45 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sde-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
# train --seed=46 --device=mps --cfg=d2d-ablate --name=sb-d2d-normal_to_moons-match_sde-fixed_var --wandb=online \
#     --overrides='data@data.p_0=d2normal,data@data.p_1=moons,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde'
