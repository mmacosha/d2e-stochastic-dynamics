#!/bin/bash

DEVICE=$1

# Ensure we're launching from the project root
cd "$(dirname "$0")/../.."

# ll matching learnt var
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-learnt_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-tv,models@models.bwd=d64-tv,sampler.matching_method=ll'

# ll matching fixed var
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_ll-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=ll'

# score matching
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_score-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_score-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_score-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_score-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_score-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=score'

# mean matching
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_mean-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_mean-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_mean-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_mean-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_mean-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=mean'

# dsbm
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm'

# dsbm++
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm++-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm++-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm++-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm++-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_dsbm++-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=dsbm++'

# sf2m
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sf2m-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sf2m-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sf2m-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sf2m-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sf2m-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sf2m'

# sde
train --seed=42 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sde-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
train --seed=43 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sde-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
train --seed=44 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sde-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
train --seed=45 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sde-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde' && \
train --seed=46 --device=${DEVICE} --cfg=d2d-ablate --name=sb-d2d-moons_to_gmm-match_sde-fixed_var --wandb=online \
    --overrides='data@data.p_0=moons,data@data.p_1=gmm,models@models.fwd=d64-ntv,models@models.bwd=d64-ntv,sampler.matching_method=sde'
